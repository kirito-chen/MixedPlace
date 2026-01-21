import os
import sys
import re
import Logger
import datetime
import random
import math

import networkx as nx
import matplotlib.pyplot as plt

from Connected import find_connected_components

from collections import defaultdict

#复制文件
import shutil

class Node:
    def __init__(self, nodeName, width, height, attribute=None):
        self.nodeName = nodeName # node的名字
        self.width = width # 宽度
        self.height = height # 高度
        # self.depth = 0 # 深度 # 目前用不到

        self.x = 0 # x坐标
        self.y = 0 # y坐标
        self.layer = 0 # 所在的层  默认都在第一层
        self.direction = "N" # 方向
        self.attribute = attribute # 可以是 terminal  terminal_NI 对应 /FIXED  /FIXED_NI
        self.moveType = None # 移动类型 /FIXED 或者 /FIXED_NI
        self.isIOMacro = False #  isIOMacro 需要进行剔除
    
    def update(self, x, y, direction, moveType = None):
        # readPL时调用，更新参数
        self.x = x
        self.y = y
        self.direction = direction
        self.moveType = moveType
    
    def modifyAttr2None(self):
        #将宏块修改成可移动
        self.attribute = None
        self.moveType = None

class Pin:
    def __init__(self, nodeName, centerOffsetX, centerOffsetY):
        self.centerOffsetX = centerOffsetX # 引脚相对于单元【中心点】的x轴偏移
        self.centerOffsetY = centerOffsetY # 引脚相对于单元【中心点】的y轴偏移
        self.offsetX = 0 # 引脚相对于单元【左下角】的偏移
        self.offsetY = 0 # 引脚相对于单元【左下角】的偏移
        self.offsetZ = 0 # 引脚是一半的单元高度 即0.5*depth
        self.nodeName = nodeName # pin所属的node
        self.layer = 0 # Pin所属的层 即pin属于的node所属的层

class Net:
    def __init__(self, netName, pinStart, partNumPin):
        self.netName = netName # net的名字
        self.pinIndex = set(range(pinStart, pinStart+partNumPin)) #存储pin下标的集合
        #如 [0 1 3]  则表示 0 1 3 引脚是该net下的

lambdaTimes = {"adaptec1": 70, "adaptec2": 160,"adaptec3": 650,"adaptec4": 460,
               "bigblue1": 120, "bigblue2": 30, "bigblue3": 470, "bigblue4": 550,
               "adaptec5": 440, "newblue1":2000, "newblue2":190, "newblue3":170, 
               "newblue4":400, "newblue5":570, "newblue6":650, "newblue7":650
              }

class DataBase:
    def __init__(self):
        #数据储存结构
        self.nodes = {} # node字典 key:nodeName value:[Node, ...] 
        self.macros = {} # macro字典 key:nodeName value:[Node, ...] 
        self.pins = [] # pin列表 所有的pin
        self.nets = {}  # net字典  key:netName  value:[Net, ...] 
         
        self.node2Net = {} # node所属net1的字典 key:nodeName value:[netName, ...]
        self.macroConnect = set() # macro以及与其相连的cell [nodeNamce,nodeName2, ...]
        self.macroConnectNets = {} # 只连接macro的net
        
        #总节点数量
        self.numNodes = 0  # node的数量 包含terminal
        self.numTerminals = 0  # terminal的数量
        self.numNets = 0 # net的数量
        self.numPins = 0 # pin的数量
        self.cutNum = 0 # 被切割的net数量

        #die属性
        self.xl = 0
        self.xh = 0
        self.yl = -1  # yl存在是0的情况
        self.yh = 0
        self.rows = [] # 存放scl文件的每一行的数据 用来清除在布局区域之外的固定宏块
        # [[subrowOrigin,numSites,coordinate,height],]  xl xh-xl yl height
        self.dieWidth = 0  # die的宽度  dieWidth = xh
        self.dieHeight = 0  # die的高度 dieHeight = xl
        self.rowHeight = 0 # 行高
        self.layerDepth = 0 # 一层的高度   层高等于行高
        self.z_max = 0 # z轴总高度 z_max = layerDepth * numLayer
        self.numRows = 0 # 行数
        self.numLayer = 0 # 分层层数
        self.reductionFactor = 0 # 缩小比例 dieWidth dieHeight numRows
        self.minMacroRowHeight = 0 # 最小的macro的行高
        
        # 计算
        self.HPWL = 0 #  2D的投影线长
        self.HPWL3d = 0 # 3D的投影线长 
        self.VI = 0 # 通孔数量
        self.macroArea = 0 # 宏块的面积和
        self.cellArea = 0 # 标准单元节点的面积和
        self.allArea = 0 # 所有层的面积
        self.averageMacroWidth = 0 # 宏块的平均宽度
        self.averageMacroHeight = 0  # 宏块的平均高度 
        
        #数据集属性
        self.auxFileName = None # 文件名 无后缀 如 adaptec1
        self.fileName = None # 文件名 无后缀有层数 如 adaptec1_2
        self.macroArea80 = 0.0 # macro area 80% 所对应的面积
        self.macroArea90 = 0.0 # macro area 90% 所对应的面积
        self.sclFile = None # scl的文件名，用于原样输出
        
    def readBookshelf(self, auxFilePath):
        #获取文件目录
        dirPath = os.path.dirname(auxFilePath)
        fileList = None
        with open(auxFilePath) as inFile:
            line = inFile.readline()
            words = line.strip().split()
            fileList = words[2:]
        for fileName in fileList:
            extension = os.path.splitext(fileName)[1] # 获取后缀名
            filePath = dirPath + '/' + fileName  # 组成文件名
            
            if extension == ".nodes":
                self.readNodes(filePath)
            elif extension == ".nets":
                self.readNet(filePath)
            elif extension == ".wts":
                self.readWts(filePath)
            elif extension == ".pl":
                self.readPlFirst(filePath)  # 第一次解析时才有资格修改movetype
            elif extension == ".scl":
                self.sclFile = filePath
                self.readScl(filePath)

    def readNodes(self, filePath):
        with open(filePath) as inFile:
            #跳过前三行
            while True:
                line = inFile.readline()
                words = line.strip().split()
                if(len(words) > 1 and words[0] == "NumNodes"):
                    self.numNodes = int(words[2])
                    break
            #读取Terminals数量
            line = inFile.readline()
            words = line.strip().split()
            self.numTerminals = int(words[2])

            while True:
                line = inFile.readline()
                if not line:
                    break
                words = line.strip().split()
                if len(words) >= 3: #std cell
                    nodeName = words[0]
                    width = int(words[1])
                    height = int(words[2])
                    attribute = None # 可以是terminal  terminalNI 对应 /FIXED  /FIXEDNI
                    if len(words) == 4:
                        attribute = words[3]
                    node = Node(nodeName, width, height, attribute)
                    self.nodes[nodeName] = node
                    if attribute == "terminal" or attribute == "terminalNI" or attribute == "terminal_NI":
                        if height >= 2:  # 排除pad
                            self.macros[nodeName] = node

    def readNet(self, filePath):
        with open(filePath) as inFile:
            #跳过前几行
            while True:
                line = inFile.readline()
                if not line:
                    break #到达文件末尾
                words = line.strip().split()
                if(len(words) > 1 and (words[0] == "UCLA" or words[0] == "#")):
                    continue 
                if len(words) < 1:
                    continue
                if words[0] == "NumNets":
                    self.numNets = int(words[2])
                elif words[0] == "NumPins":
                    self.numPins = int(words[2])
                elif words[0] == "NetDegree":
                    partNumPin = int(words[2])
                    netName = words[3]
                    net = Net(netName, len(self.pins), partNumPin)
                    self.nets[netName] = net
                    for j in range(partNumPin):
                        line = inFile.readline()
                        words = line.strip().split()
                        if len(words) != 5:
                            print("readNet ERROR! pin is not correct")
                            exit(1)
                        nodeName = words[0]
                        centerX = float(words[3])
                        centerY = float(words[4])
                        pin = Pin(nodeName, centerX, centerY)
                        self.pins.append(pin)

    def readWts(self, filePath):
        pass #该文件为空，不用读取
    
    def readPlFirst(self, filePath):
        with open(filePath) as inFile:
            while True:
                line = inFile.readline()
                if not line:
                    break  # 文件末尾
                words = line.strip().split()
                #跳过前几行
                if(len(words) > 1 and (words[0] == "UCLA" or words[0] == "#")):
                    continue
                if len(words) >= 5:
                    nodeName = words[0]
                    x = float(words[1])
                    y = float(words[2])
                    dire = words[4]
                    moveType = None
                    if len(words) == 6:
                        moveType = words[5]
                    node = self.nodes[nodeName]
                    node.update(x, y, dire, moveType)
    
    def readPl(self, filePath):
        #只修改坐标
        with open(filePath) as inFile:
            while True:
                line = inFile.readline()
                if not line:
                    break  # 文件末尾
                words = line.strip().split()
                #跳过前几行
                if(len(words) > 1 and (words[0] == "UCLA" or words[0] == "#")):
                    continue
                if len(words) >= 5:
                    nodeName = words[0]
                    # 跳过虚拟宏块
                    if nodeName == "v1":
                        continue
                    x = float(words[1])
                    y = float(words[2])
                    node = self.nodes[nodeName]
                    node.x = x
                    node.y = y 
                    
    def readKahyparMacro(self, kahyparPartitionFile):
        with open(kahyparPartitionFile, "r", encoding="utf-8") as f:
            for line in f:
                words = line.strip().split()
                nodeName = words[0]
                layer = int(words[1])
                node = self.nodes[nodeName]
                if node.attribute == "terminal":
                    #只有宏才改变层数
                    node.layer = layer

    def readScl(self, filePath):
        # 只读取第一行的数据 不行
        # 每一行会不一样，得读取全部数据
        with open(filePath) as inFile:
            maxX = 0
            minX = sys.maxsize
            while True:
                line = inFile.readline()
                if not line:
                    break #到达文件末尾
                if line.strip() == "":
                    continue
                words = line.strip().split()
                if words[0] == "NumRows":
                    self.numRows = int(words[2])
                    for i in range(self.numRows):
                        subrowOrigin,numSites,coordinate,height = 0,0,0,0
                        while True:
                            line = inFile.readline()
                            if line.strip() == "":
                                continue
                            words = line.strip().split()
                            if words[0] == "Coordinate":
                                coordinate = int(words[2])
                            elif words[0] == "Height":
                                height = int(words[2])
                            elif words[0] == "Sitewidth":
                                sitewidth = int(words[2])  #为了适应数据集openroad， Sitewidth不为1
                            elif words[0] == "SubrowOrigin":
                                subrowOrigin = int(words[2])
                                numSites = int(words[5])
                                break
                        self.rows.append([subrowOrigin,numSites,coordinate,height])
                        if subrowOrigin + numSites * sitewidth > maxX:
                            maxX = subrowOrigin + numSites * sitewidth
                        if minX > subrowOrigin:
                            minX = subrowOrigin
                            
                        if self.rowHeight == 0:
                            self.rowHeight = height
                        if self.yl == -1:
                            self.yl = coordinate
                    
                    self.xh = maxX
                    self.xl = minX
                    #计算dieHeight = 行数 * 行高
                    self.dieWidth = self.xh - self.xl
                    self.dieHeight = self.numRows * self.rowHeight
                    self.yh = self.dieHeight + self.yl
                    break
                           
    def generateJson(self, outputDir, fileName, auxPath, resultDir, numLayer, curlayer = -1, gpu = 0, gp = 0, lg = 1, dp = 1,
                      plotF = 0, _stopOverflow = 0.03, _enable_fillers = 1, onlyMacroLgFlag = 0, _numBins = 512, _num_bins_z = 64,
                      _deterministic_flag = 1, _random_seed = 1000, _iteration = 1000, _zAlpha = 1.0, _num_threads = 40):
        """ 生成dreamplace需要的json配置文件
            stopOverflow: 0.03 or 0.07
            randomFlag: 0 or 1 or 2  1 是正态随机分布
            result_dir: dreamplace_temp or dreamplace_temp_cell
        """
        # json/
        if not os.path.isdir(outputDir):  
            Logger.printPlace(outputDir + " folder does not exist, create folder")
            os.system("mkdir -p %s" % (outputDir))
        # json/adaptec1
        fileNameDir = os.path.join(outputDir, fileName) 
        if not os.path.isdir(fileNameDir):
            Logger.printPlace(fileNameDir + " folder does not exist, create folder")
            os.system("mkdir -p %s" % (fileNameDir))
        fileNameSecondDir = fileNameDir
        if curlayer != -1:
            # json/adaptec1/layer_1
            fileNameSecondDir = os.path.join(fileNameDir, "layer_"+str(curlayer))
            if not os.path.isdir(fileNameSecondDir):
                Logger.printPlace(fileNameSecondDir + " folder does not exist, create folder")
                os.system("mkdir -p %s" % (fileNameSecondDir))
        
        # 2d与3d有区别的参数
        gpuFlag = gpu
        gpFlag = gp
        legalFlag = lg
        dpFlag = dp
        plotFlag = plotF # 2D的情况下可以打开
        zAlpha = _zAlpha
        # 固定参数
            
        
        # 可调整优化的参数
        # numBins = 512  # x 和 y的bin大小
        # num_bins_z = 8 # z方向bin大小 默认为64
        numBins = _numBins  # x 和 y的bin大小
        num_bins_z = _num_bins_z # z方向bin大小 默认为8
        iteration = _iteration  # 迭代次数
        wirelength = "weighted_average"
        optimizer = "nesterov" # ["nesterov", "adam"]
        
        enable_fillers = _enable_fillers
        stopOverflow = _stopOverflow
        randomFlag = 1
        deterministic_flag = _deterministic_flag
        # deterministic_flag = 0 # 新加的
        random_seed = _random_seed
        # random_seed = 199 # 新加的
        num_threads = _num_threads #40  # 设置成单线程
        
        #画图 也不用开 ISPD画不了
        draw_mode = "3D"  # 调用的画图方式，有"3D"与"3DCube"可选
        draw_iter_interval = 20 # 前500次迭代，间隔多少次进行画图
        layer_plot_flag = 0


        lineStr = "{\n"
        lineStr += "    \"aux_input\": \"{}\",\n".format(auxPath)
        lineStr += "    \"gpu\": {},\n".format(gpuFlag)
        lineStr += "    \"num_bins_x\": {},\n".format(numBins)
        lineStr += "    \"num_bins_y\": {},\n".format(numBins)
        lineStr += "    \"global_place_stages\": [\n"
        lineStr += "        { "
        lineStr +=             "\"num_bins_x\": {}, \"num_bins_y\": {}, ".format(numBins, numBins)
        lineStr +=             "\"iteration\": {}, \"learning_rate\": {}, ".format(iteration, 0.01)
        lineStr +=             "\"wirelength\": \"{}\", \"optimizer\": \"{}\", ".format(wirelength, optimizer)
        lineStr +=             "\"Llambda_density_weight_iteration\": {}, ".format(1)
        lineStr +=             "\"Lsub_iteration\": {} ".format(1)
        lineStr +=         "}\n"
        lineStr += "    ],\n"
        lineStr += "    \"target_density\": {},\n".format(1)
        lineStr += "    \"density_weight\": {},\n".format("8e-5")
        lineStr += "    \"gamma\": {},\n".format("4.0")
        lineStr += "    \"random_seed\": {},\n".format(_random_seed)
        lineStr += "    \"scale_factor\": {},\n".format("1.0")
        lineStr += "    \"ignore_net_degree\": {},\n".format("100")
        lineStr += "    \"enable_fillers\": {},\n".format(enable_fillers)
        lineStr += "    \"gp_noise_ratio\": {},\n".format("0.025")
        lineStr += "    \"global_place_flag\": {},\n".format(gpFlag)
        lineStr += "    \"legalize_flag\": {},\n".format(legalFlag)
        lineStr += "    \"detailed_place_flag\": {},\n".format(dpFlag)
        lineStr += "    \"detailed_place_engine\": \"\",\n"
        lineStr += "    \"detailed_place_command\": \"\",\n"
        lineStr += "    \"stop_overflow\": {},\n".format(stopOverflow)
        lineStr += "    \"dtype\": \"{}\",\n".format("float32")
        lineStr += "    \"plot_flag\": {},\n".format(plotFlag)
        lineStr += "    \"random_center_init_flag\": {},\n".format(randomFlag)
        lineStr += "    \"sort_nets_by_degree\": {},\n".format(0)
        lineStr += "    \"num_threads\": {},\n".format(num_threads)
        lineStr += "    \"deterministic_flag\": {},\n".format(deterministic_flag)
        lineStr += "    \"result_dir\": \"{}\",\n".format(resultDir)
        # place 2D单独宏块合法化需要用到 macroLgFlag
        lineStr += "    \"onlyMacroLgFlag\": {},\n".format(onlyMacroLgFlag)
        
        # place 3D需要用到的
        lineStr += "    \"z_max\": {},\n".format(self.z_max) # z轴最大高度 z_max = self.layerDepth * self.numLayer
        lineStr += "    \"layerDepth\": {},\n".format(self.layerDepth) # 一层层高 也就是单元高度
        lineStr += "    \"num_bins_z\": {},\n".format(num_bins_z)
        lineStr += "    \"layer\": {},\n".format(numLayer)  # 告诉3D需要划分的层次
        lineStr += "    \"draw_mode\": \"{}\",\n".format(draw_mode)
        lineStr += "    \"draw_iter_interval\": {},\n".format(draw_iter_interval)
        lineStr += "    \"layer_plot_flag\": {},\n".format(layer_plot_flag)
        lineStr += "    \"zAlpha\": {}\n".format(zAlpha)
        lineStr += "}"
        filePath = os.path.join(fileNameSecondDir, fileName+".json")
        with open(filePath,"w",encoding="utf-8") as f:
            f.write(lineStr)
        return filePath

    def parseLayerFile(self, layerFile:str):
        with open(layerFile, "r", encoding="utf-8") as f:
            for line in f:
                words = line.strip().split()
                nodeName = words[0]
                layer = int(words[1])
                node = self.nodes[nodeName]
                node.layer = layer
        
        #将pin的layer也赋值
        for pin in self.pins:
            # 有的pin已经被删除了
            if pin.nodeName in self.nodes:
                node = self.nodes[pin.nodeName]
                pin.layer = node.layer
    
    def parsezPosFile(self, zPosFile):
        # 只有在cell 3DGP后才会使用这部分 确保面积不会溢出
        Logger.printPlace(f"Parsing result file " + zPosFile  + " ...")
        # 读取zPos 具体z轴位置
        nodezPos = {}
        with open(zPosFile, "r", encoding="utf-8") as f:
            for line in f:
                words = line.strip().split()
                nodeName = words[0]
                zPos = float(words[1])
                nodezPos[nodeName] = zPos
                

        dieArea = self.dieWidth * self.dieHeight * 0.995
        layerArea = [0] * (self.numLayer + 1)
        # 统计宏块面积
        for _, macro in self.macros.items():
            # 宏块可能不在行上
            # 宏块占用的那一行不要
            yl = int(macro.y/self.rowHeight) * self.rowHeight  # 向下取整
            yh = math.ceil((macro.y + macro.height)/self.rowHeight) * self.rowHeight # 向上取整
            macroArea = macro.width * (yh - yl)
            layerArea[macro.layer] += macroArea
            #
        # print(f"dieArea:{dieArea}")
        # print(layerArea)
        # 判断每层面积
        for nodeName, zPos in nodezPos.items():
            node = self.nodes[nodeName]
            nodeArea = node.width * node.height 
            # 面积过大放在其他层
            if nodeArea + layerArea[node.layer] > dieArea:
                # print(f"debug:area is not enough, {nodeName} is not in layer {node.layer}")
                # 计算每一层的实际距离
                layerDistance = []
                for i in range(self.numLayer):
                    layerDistance.append((i,abs(zPos - i*self.layerDepth)))
                layerDistanceSorted = sorted(layerDistance, key=lambda x: x[1])
                for layer,_ in layerDistanceSorted:
                    if nodeArea + layerArea[layer] <= dieArea:
                        node.layer = layer
                        # print(f"debug:{nodeName} is set to layer {node.layer}")
                        break
            # 确定放置在该层
            layerArea[node.layer] += nodeArea
    
    def generateBookShelf(self, outputDir:str, fileName:str, curlayer:int = -1, moveMacro = False, originSclFile = False):
        """ 第一个参数表示一级文件夹 temp
            第二个参数表示名字 adaptec1
            则形成路径   outputDir/fileName/layer_1/fileName.xx   
            eg.  bookshelf/adaptec1/layer_1/adaptec1.aux
            第三个参数 curlayer 表示生成的文件所在层数, 只在2d时用得到(mix2d,cell2d)
            第四个参数 moveMacro表示是否可以移动宏块, 在Mixed环节需要用到(mix3d,mix2d)
        """
        # bookshelf/
        if not os.path.isdir(outputDir):  
            Logger.printPlace(outputDir + " folder does not exist, create folder")
            os.system("mkdir -p %s" % (outputDir))
        # bookshelf/adaptec1
        fileNameDir = os.path.join(outputDir, fileName) 
        if not os.path.isdir(fileNameDir):
            Logger.printPlace(fileNameDir + " folder does not exist, create folder")
            os.system("mkdir -p %s" % (fileNameDir))
        
        fileNameSecondDir = fileNameDir # 不分层的话，用两层目录就够了，不需要layer级别
        if curlayer != -1:
            # bookshelf/adaptec1/layer_1
            fileNameSecondDir = os.path.join(fileNameDir, "layer_"+str(curlayer))
            if not os.path.isdir(fileNameSecondDir):
                Logger.printPlace(fileNameSecondDir + " folder does not exist, create folder")
                os.system("mkdir -p %s" % (fileNameSecondDir))
            
        auxFile = self.generateAux(fileNameSecondDir, fileName)
        self.generateWts(fileNameSecondDir, fileName)
        self.generateScl(fileNameSecondDir, fileName, originSclFile)
        self.generateNets(fileNameSecondDir, fileName)
        
        nodeInThisLayer = True # 分层条件下可能为False 即该层没有可移动节点 则应该跳过
        
        if curlayer != -1: # 需要分层，2d情况下。mix2d时moveMacro为True，cell2d时为False 
            nodeInThisLayer = self.generateNodesMultilTier(fileNameSecondDir, fileName, curlayer, moveMacro)
            self.generatePlMultilTier(fileNameSecondDir, fileName, curlayer, moveMacro)
        else:  # 不需要分层 3d情况下。mix3d时moveMacro为True， cell3d时为False
            self.generateNodes(fileNameSecondDir, fileName, moveMacro)
            self.generatePl(fileNameSecondDir, fileName, moveMacro)  
        return auxFile, nodeInThisLayer
    
    def generateAux(self, outputFileDir, fileName):
        filePath = os.path.join(outputFileDir, fileName+".aux")
        with open(filePath,"w",encoding="utf-8") as f:
            lineStr="RowBasedPlacement :  {0}.nodes  {0}.nets  {0}.wts  {0}.pl  {0}.scl".format(fileName)
            f.write(lineStr)
        return filePath

    def generateNodes(self, outputFileDir, fileName, moveMacro):
        headerStr="UCLA nodes 1.0\n"
        headerStr+="# Created  : {}\n".format(datetime.datetime.now().strftime('%Y %m %d'))
        headerStr+="# User     : cjq\n\n"
        numberStr=""
        entriesStr=""
        terminalStr=""
        numTerminal = 0
        numNodes = 0
        if moveMacro:
            for nodeName, node in self.nodes.items():
                entriesStr+=" {}  {}  {}\n".format(nodeName, node.width, node.height)  # 全部不赋值terminal
                numNodes += 1
        else:
            for nodeName, node in self.nodes.items():
                if node.attribute == None:
                    entriesStr+=" {}  {}  {}\n".format(nodeName, node.width, node.height)
                else:
                    terminalStr+=" {}  {}  {}  {}\n".format(nodeName, node.width, node.height, node.attribute)
                    numTerminal += 1
                numNodes += 1
                    
        numberStr+="NumNodes :   {}\n".format(numNodes)
        numberStr+="NumTerminals :   {}\n\n".format(numTerminal)
        filePath = os.path.join(outputFileDir, fileName+".nodes")
        with open(filePath,"w",encoding="utf-8") as f:
            f.write(headerStr)
            f.write(numberStr)
            f.write(entriesStr)
            if terminalStr != "":
                f.write(terminalStr)
        Logger.printPlace("Generate new node file " + filePath)
    
    def generateNodesMultilTier(self, outputFileDir, fileName, layer, moveMacro):
        #考虑所有层的节点
        headerStr="UCLA nodes 1.0\n"
        headerStr+="# Created  : {}\n".format(datetime.datetime.now().strftime('%Y %m %d'))
        headerStr+="# User     : cjq\n\n"
        numberStr=""
        entriesStr=""
        terminalStr=""
        terminalNIStr=""
        numTerminal = 0
        numTerminalNI = 0
        nodeInThisLayer = False
        for nodeName, node in self.nodes.items():
            if node.layer == layer:
                if node.attribute == None or (moveMacro and nodeName != "v1"):
                    # moveMacro 为True则一直输出为None的情况
                    nodeInThisLayer = True
                    entriesStr += " {}  {}  {}\n".format(nodeName, node.width, node.height)
                elif node.attribute == "terminal":
                    terminalStr += " {}  {}  {}  terminal\n".format(nodeName, node.width, node.height)
                    numTerminal += 1
                else: #terminal_NI
                    terminalNIStr += " {}  {}  {}  terminal_NI\n".format(nodeName, node.width, node.height)
                    numTerminalNI += 1
                    
            else:
                #不在同一层的都修改为terminal_NI
                terminalNIStr += " {}  {}  {}  terminal_NI\n".format(nodeName, node.width, node.height)
                numTerminalNI += 1       
                    
        numberStr+="NumNodes :   {}\n".format(self.numNodes)
        numberStr+="NumTerminals :   {}\n\n".format(numTerminal + numTerminalNI)
        filePath = os.path.join(outputFileDir, fileName+".nodes")
        with open(filePath,"w",encoding="utf-8") as f:
            f.write(headerStr)
            f.write(numberStr)
            f.write(entriesStr)
            f.write(terminalStr)
            f.write(terminalNIStr)
        return nodeInThisLayer

    def generateNets(self, outputFileDir, fileName):
        # 目前只考虑当前层的net 
        headerStr="UCLA nets 1.0\n"
        headerStr+="# Created  : {}\n".format(datetime.datetime.now().strftime('%Y %m %d'))
        headerStr+="# User     : cjq\n\n"
        numberStr=""
        entriesStr=""
        numNets = 0
        numPins = 0
        for netName, net in self.nets.items():
            netPinEntries = ""
            numNets += 1
            for i in net.pinIndex:
                numPins += 1
                pin = self.pins[i]
                netPinEntries+="  {}  {} :  {}  {}\n".format(pin.nodeName, "I", format(pin.centerOffsetX,"0.6f"),format(pin.centerOffsetY,"0.6f"))
            entriesStr += "NetDegree : {}   {}\n".format(len(net.pinIndex), net.netName)
            entriesStr += netPinEntries

        numberStr+="NumNets : {}\n".format(numNets)
        numberStr+="NumPins : {}\n\n".format(numPins)
        filePath = os.path.join(outputFileDir, fileName+".nets")
        with open(filePath,"w",encoding="utf-8") as f:
            f.write(headerStr)
            f.write(numberStr)
            f.write(entriesStr)

    def generateWts(self, outputFileDir, fileName):
        headerStr="UCLA wts 1.0\n"
        headerStr+="# Created  : {}\n".format(datetime.datetime.now().strftime('%Y %m %d'))
        headerStr+="# User     : cjq\n\n"
        numberStr=""
        entriesStr=""
        filePath = os.path.join(outputFileDir, fileName+".wts")
        with open(filePath,"w",encoding="utf-8") as f:
            f.write(headerStr)
            f.write(numberStr)
            f.write(entriesStr)

    def generatePl(self, outputFileDir, fileName, moveMacro):
        headerStr="UCLA pl 1.0\n"
        headerStr+="# Created  : {}\n".format(datetime.datetime.now().strftime('%Y %m %d'))
        headerStr+="# User     : cjq\n\n"
        entriesStr=""
        terminalStr=""
        if moveMacro:
            for nodeName, node in self.nodes.items():
                entriesStr+=" {}  {}  {} : {}\n".format(nodeName, format(node.x,"0.4f"), format(node.y,"0.4f"), node.direction)
        else:        
            for nodeName, node in self.nodes.items():
                if node.moveType == None:
                    entriesStr+=" {}  {}  {} : {}\n".format(nodeName, format(node.x,"0.4f"), format(node.y,"0.4f"), node.direction)
                else:
                    terminalStr+=" {}  {}  {} : {} {}\n".format(nodeName, format(node.x,"0.4f"), format(node.y,"0.4f"), node.direction, node.moveType)
        
        filePath = os.path.join(outputFileDir, fileName+".pl")
        with open(filePath,"w",encoding="utf-8") as f:
            f.write(headerStr)
            f.write(entriesStr)
            if terminalStr != "":
                f.write(terminalStr)
            
    def generatePlMultilTier(self, outputFileDir, fileName, layer, moveMacro):
        headerStr="UCLA pl 1.0\n"
        headerStr+="# Created  : {}\n".format(datetime.datetime.now().strftime('%Y %m %d'))
        headerStr+="# User     : cjq\n\n"
        entriesStr=""
        fixedStr=""
        fixedNIStr=""
        
        for nodeName, node in self.nodes.items():
            if node.layer == layer:
                if node.moveType == None or (moveMacro and nodeName != "v1"):
                    # moveMacro 为真则一直输出为None的情况，让macro可移动
                    entriesStr+=" {}  {}  {} : {}\n".format(nodeName, node.x, node.y, node.direction)
                elif node.moveType == "/FIXED":
                    fixedStr+=" {}  {}  {} : {} /FIXED\n".format(nodeName, node.x, node.y, node.direction)
                else:
                    fixedNIStr+=" {}  {}  {} : {} /FIXED_NI\n".format(nodeName, node.x, node.y, node.direction)
            else:
                #不在同一层的都修改为 /FIXED_NI
                fixedNIStr+=" {}  {}  {} : {} /FIXED_NI\n".format(nodeName, node.x, node.y, node.direction)
        
        filePath = os.path.join(outputFileDir, fileName+".pl")
        with open(filePath,"w",encoding="utf-8") as f:
            f.write(headerStr)
            f.write(entriesStr)
            f.write(fixedStr)
            f.write(fixedNIStr)

    def generateScl(self, outputFileDir, fileName, tooManyMacro = False, originSclFile = False):
        if originSclFile:
            filePath = os.path.join(outputFileDir, fileName+".scl")
            shutil.copyfile(self.sclFile, filePath)
            return True
    
        headerStr="UCLA scl 1.0\n"
        headerStr+="# Created  : {}\n".format(datetime.datetime.now().strftime('%Y %m %d'))
        headerStr+="# User     : cjq\n\n"
        numberStr=""
        entriesStr=""
        
        numRows = self.numRows
        rowHeight = self.rowHeight
        # 2D LG时才会出现
        if tooManyMacro:
            # 找到最小的宏块的行高，以此作为一行的高度，并重新计算行数
            minHeight = sys.maxsize
            for macroName, macro in self.macros.items():
                if minHeight > macro.height:
                    minHeight = macro.height
            rowHeight = minHeight
            # 重新计算行数
            numRows = int(self.dieHeight / rowHeight)
        self.minMacroRowHeight = rowHeight
        

        numberStr+="NumRows : {}\n\n".format(numRows)
        coordinate = 0
        for i in range(numRows):
            entriesStr += "CoreRow Horizontal\n"
            entriesStr += "  Coordinate    :    {}\n".format(coordinate)
            entriesStr += "  Height        :    {}\n".format(rowHeight)
            entriesStr += "  Sitewidth     :    {}\n".format(1)
            entriesStr += "  Sitespacing   :    {}\n".format(1)
            entriesStr += "  Siteorient    :    {}\n".format(1)
            entriesStr += "  Sitesymmetry  :    {}\n".format(1)
            entriesStr += "  SubrowOrigin  :    {}	NumSites  :  {}\n".format(0, self.dieWidth) # 因为从0开始计算，所以宽度只有dieWidth
            entriesStr += "End\n"
            coordinate += rowHeight

        filePath = os.path.join(outputFileDir, fileName+".scl")
        with open(filePath,"w",encoding="utf-8") as f:
            f.write(headerStr)
            f.write(numberStr)
            f.write(entriesStr)

    def calculHPWL(self):
        # 必须先执行一次updatePinOffset
        # 计算半周线长  xyz每个维度都找最大的差值
        layerDepth = self.layerDepth # 行高 一层的深度 
        HPWL3d = 0.0 # 3d 的投影线长
        HPWL = 0.0  # 2d 的投影线长
        VI = 0 # 通孔数量 如：layer1 到layer3 为2
        for netName, net in self.nets.items():
            max_x,max_y,max_z,min_x,min_y,min_z = 0,0,0,float("inf"),float("inf"),float("inf")
            min_layer = 100
            max_layer = 0
            for i in net.pinIndex:
                pin = self.pins[i]
                node = self.nodes[pin.nodeName]
                x = node.x + pin.offsetX 
                y = node.y + pin.offsetY
                z = node.layer * layerDepth + pin.offsetZ
                
                max_x = max(max_x, x)
                max_y = max(max_y, y)
                max_z = max(max_z, z)
                
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                min_z = min(min_z, z)
                
                min_layer = min(min_layer, node.layer)
                max_layer = max(max_layer, node.layer)
            partHPWL3d = max_x - min_x + max_y - min_y + max_z - min_z
            partHPWL =  max_x - min_x + max_y - min_y
            HPWL3d += partHPWL3d
            HPWL += partHPWL
            VI += max_layer - min_layer
        self.HPWL = int(HPWL)
        self.HPWL3d = int(HPWL3d)
        self.VI = VI
        # print(f"HPWLZ:{HPWLZ}")

    def readResultFile(self, plFile, layerFile, curPhase, zPosFile = None):
        #读入pl和layer文件
        Logger.printPlace(f"Parsing {curPhase} result file " + plFile  + " ...")
        self.readPl(plFile)
        if layerFile != None:
            Logger.printPlace(f"Parsing {curPhase} result file " + layerFile  + " ...")
            self.parseLayerFile(layerFile)
        
    def randomFixedMacro(self):
        # 随机对macro分层并随机位置
        area = [0] * self.numLayer # 记录每一层的面积
        for nodeName, node in self.nodes.items():
            if node.attribute == "terminal":
                # 找到面积最小的两层，并随机选择一层作为宏块的放置层数
                sorted_numbers = sorted(enumerate(area), key=lambda x: x[1])
                smallest_indices = [index for index, value in sorted_numbers[:2]]
                placeLayer = random.choice(smallest_indices)
                node.layer = placeLayer
                # x在[0, self.dieWidth-node.width]随机  y在[0,(self.dieHeight-node.height)/self.rowHeight] *self.rowHeight中取值
                node.x = random.randint(0, self.dieWidth-node.width)
                node.y = random.randint(0,(self.dieHeight-node.height)/self.rowHeight)*self.rowHeight
        # 对pin的layer也进行修改
        for pin in self.pins:
            node = self.nodes[pin.nodeName]
            pin.layer = node.layer

    def updateMacroConnectNets(self):
        netInMacroConnect = [] # 记录由macro以及其连接的单元组成的net
        nodeNameStr = set()
        for netName, net in self.nets.items():
            numPinInMacroConnect = 0
            nodeNameSet = []
            for i in net.pinIndex:
                # 先遍历一遍判断该net是否连接两个以上的单元
                pin = self.pins[i]
                if pin.nodeName in self.macroConnect and pin.nodeName not in nodeNameSet:
                    # 去除重复的
                    nodeNameSet.append(pin.nodeName)
                    numPinInMacroConnect += 1
            nodeNameSetSort = sorted(nodeNameSet) # 有序
            # 将nodeName变成string
            nameStr = ""
            for nodeName in nodeNameSetSort:
                nameStr += nodeName
            
            if numPinInMacroConnect >= 2 and nameStr not in nodeNameStr:
                netInMacroConnect.append(netName)
                nodeNameStr.add(nameStr)
        
        for netName in netInMacroConnect:
            net = self.nets[netName]
            macroNet = Net(netName,0,0)
            nodeNameSet = set()
            for i in net.pinIndex:
                # 第二次遍历 拿到net当中的包含在macro的单元
                pin = self.pins[i]
                # if pin.nodeName in self.macroConnect：
                if pin.nodeName in self.macroConnect and pin.nodeName not in nodeNameSet:
                    # 去除重复的
                    nodeNameSet.add(pin.nodeName)
                    macroNet.pinIndex.add(i)
            self.macroConnectNets[netName] = macroNet
        # more
        # 考虑macro1 - cell - macro2的情况，需要生成边macro1 - macro2
        macro2macro = {} # macro与其相连的macro key:name value:{name1,name2}
        for netName,net in self.macroConnectNets.items():
            nodeNameList = []
            for i in net.pinIndex:
                nodeName = self.pins[i].nodeName
                nodeNameList.append(nodeName)
            for nodeName in nodeNameList:
                if nodeName not in macro2macro:
                    macro2macro[nodeName] = set() # 独一无二
                for nodeName2 in nodeNameList:
                    if nodeName != nodeName2:
                        macro2macro[nodeName].add(nodeName2)
        # 孤立的macro
        for macroName, macro in self.macros.items():
            if macroName not in macro2macro:
                macro2macro[macroName] = set() # 独一无二

        # 寻找只有一个macro的net
        newNetNum = 1
        for netName, net in self.nets.items():
            macroNameSet = set()
            macroPin = set()
            for i in net.pinIndex:
                # 判断是否是宏块
                if self.pins[i].nodeName in self.macros:
                    macroNameSet.add(self.pins[i].nodeName)
                    macroPin.add(i)
            # 只有一个宏块 则判断其他点是否连接到其他宏块 只考虑一层
            
            
            if len(macroNameSet) == 1:
                macroName = list(macroNameSet)[0]
                for i in net.pinIndex:
                    # 跳过
                    if i in macroPin:
                        continue
                    nodeName = self.pins[i].nodeName
                    netList = self.node2Net[nodeName]
                    for netName2 in netList:
                        # 跳过
                        if netName2 == netName:
                            continue 
                        net2 = self.nets[netName2]
                        for j in net2.pinIndex:
                            nodeName2 = self.pins[j].nodeName
                            if nodeName2 == macroName:
                                continue
                            if nodeName2 in self.macros and nodeName2 not in macro2macro[macroName] and j not in macroPin:
                                macro2macro[macroName].add(nodeName2)
                                macro2macro[nodeName2].add(macroName)
                                # 添加一条边
                                newNetName = "vn" + str(newNetNum)
                                newNetNum += 1
                                newNet = Net(newNetName,0,0)
                                # 随便取一个macro的结点连接
                                macroPin = list(macroPin)
                                newNet.pinIndex.add(macroPin[0])
                                newNet.pinIndex.add(j)  
                                self.macroConnectNets[newNetName] = newNet

    def updateData(self, numbins_x, numbins_y, numbins_z):
        
        
        '''
        ##########################
        # 去掉不在放置区域内的宏块 #
        ##########################
        besideRegion = set() # 在放置区域之外的宏块
        # 不去除一半在放置区域内的
        for macroName,macro in self.macros.items():
            xLeft = macro.x
            yBottom = macro.y
            xRight = macro.x + macro.width
            yUp = macro.y + macro.height
            if xLeft > self.xh or xRight < self.xl or yBottom > self.yh or yUp < self.yl:
                # 满足条件，该宏块需要去除
                besideRegion.add(macroName)
                continue
                
            if yBottom >= self.yl:
                # 找到所在行
                rowIndex = int((yBottom - self.yl) / self.rowHeight)
                subrowOrigin,numSites,coordinate,height = self.rows[rowIndex]
                xl = subrowOrigin
                xh = subrowOrigin + numSites
                if xLeft > xh or xRight < xl:
                    # 满足条件，该宏块需要去除
                    besideRegion.add(macroName)
                    
        # 先清理net
        besideNet = [] #需要删除的边
        for netName, net in self.nets.items():
            besidePin = []
            #先遍历
            for i in net.pinIndex:
                pin = self.pins[i]
                if pin.nodeName in besideRegion:
                    besidePin.append(i)
            #再删除
            if len(net.pinIndex) - len(besidePin) > 1 : #所剩下的点要大于1才作为边  !!! 原本的数据集中是存在一个net中只存在一个pin的情况的
                for i in besidePin:
                    net.pinIndex.remove(i)
            else:
                #需要整个边进行删除
                besideNet.append(netName)
        for netName in besideNet:
            del self.nets[netName]
        # 删除nodes macros  
        # pins不用删除，不影响generateBookshelf
        for macroName in besideRegion:
            del self.nodes[macroName]
            del self.macros[macroName]
        # 更新数量
        self.numNodes = len(self.nodes)
        self.numTerminals = len(self.macros)
        self.numNets = len(self.nets)
        numPins = 0
        for _, net in self.nets.items():
            numPins += len(net.pinIndex)
        self.numPins = numPins
        '''


        ###########
        # 缩小版图 #
        ###########
        
        multip = 1.1 # 1.1 
        if self.auxFileName == "bigblue3" or self.auxFileName == "newblue3" or self.auxFileName == "newblue1": # ePlace3d 提到的  BIGBLUE3, NEWBLUE2 and NEWBLUE3
            multip = self.numLayer * 0.1 + 1
        
        reductionFactor = math.sqrt(multip /self.numLayer)    
        # reductionFactor = 1
        Logger.printPlace(f"multip: {multip}")
        Logger.printPlace(f"original dieWidth: {self.dieWidth} , dieHeight: {self.dieHeight}")
        self.dieWidth = int(self.dieWidth * reductionFactor)
        self.dieHeight = int(self.dieHeight * reductionFactor)
        self.numRows = int(self.dieHeight / self.rowHeight)
        self.dieHeight = self.numRows * self.rowHeight # 丢掉不成行的部分
        # self.xl self.xh self.yl self.yh 废弃使用
        self.xl = None
        self.xh = None
        self.yl = None
        self.yh = None
        Logger.printPlace(f"After zooming out: dieWidth {self.dieWidth} , dieHeight {self.dieHeight}")
        
        
        ##################
        # 设定单元z轴深度 #
        ##################
        
        # beta_z = _beta_z
        # if beta_z == -1:
        #     beta_z = self.numLayer / self.numRows * 100
        # # beta_z = self.numLayer / self.numRows * 100
        # Logger.printPlace(f"beta_z: {beta_z}")
        # self.z_max = int(max(self.dieWidth,self.dieHeight) * beta_z)

        # D2D 的复制方式， z_max/Nz = (x_max/Nx + y_max/Ny)/2
        self.z_max = int((self.dieWidth/ numbins_x + self.dieHeight/ numbins_y) * numbins_z / 2)

        self.layerDepth = int(self.z_max / self.numLayer)
        
        
        Logger.printPlace(f"zh: {self.z_max} , layerDepth: {self.layerDepth}")
        # exit()
        # self.layerDepth = int((self.dieWidth + self.dieHeight) / (2 * self.numLayer)) # 方案1 单元高度为平均高度
        
        #########################
        # 更新pin相对左下角的位置 #
        #########################
        for pin in self.pins:
            node = self.nodes[pin.nodeName]
            pin.offsetX = pin.centerOffsetX + node.width / 2 
            pin.offsetY = pin.centerOffsetY + node.height / 2 
            pin.offsetZ = self.layerDepth / 2
        
        ################   
        # 更新node2net #
        ################
        for netName,net in self.nets.items():
            for i in net.pinIndex:
                pin = self.pins[i]
                nodeName = pin.nodeName
                if nodeName in self.node2Net:
                    self.node2Net[nodeName].append(netName)
                else:
                    _list = []
                    _list.append(netName)
                    self.node2Net[nodeName] = _list 

        # 统计宏块与其相联的模块的数量
        '''
        macroConnect = set()
        for macroName, macro in self.macros.items():
            #遍历每个宏块
            macroConnect.add(macroName)
            netList = self.node2Net[macroName]
            for netName in netList:
                #遍历宏块相连的每条边
                net = self.nets[netName]
                for i in net.pinIndex:
                    #遍历边的每个点
                    pin = self.pins[i]
                    nodeName = pin.nodeName
                    macroConnect.add(nodeName)
        self.macroConnect = macroConnect
        '''
        # 直接以macro的key作为目标
        self.macroConnect = list(self.macros.keys())
        
        #########################
        # 统计cell面积与宏块面积 #
        #########################
        macroArea = 0 # 宏块的面积和
        nodeArea = 0 # 所有节点的面积和
        for _, macro in self.macros.items():
            macroArea += macro.width * macro.height
            
        for _, node in self.nodes.items():
            nodeArea += node.width * node.height
        self.macroArea = macroArea
        self.cellArea = nodeArea - macroArea
        self.allArea = self.dieWidth * self.dieHeight * self.numLayer 
        
        ###############################
        # 计算macro的平均宽度和平均高度 #
        ###############################
        widthAll = 0
        heightAll = 0
        for _, macro in self.macros.items():
            widthAll += macro.width
            heightAll += macro.height
        self.averageMacroWidth = int(widthAll / len(self.macros))
        self.averageMacroHeight = int(heightAll / len(self.macros))
        
        #################################
        # 计算宏块面积80% 90%所对应的面积 #
        #################################
        area2num = {} # key=area value = [num, accNum]  面积 数量 累计数量
        for _, macro in self.macros.items():
            area = macro.width * macro.height
            if area in area2num:
                area2num[area][0] += 1
            else:
                area2num[area] = [1, 0] 
        sortedArea2num = {key: area2num[key] for key in sorted(area2num.keys())}
        allNum = 0
        for area,[num,_] in sortedArea2num.items():
            allNum += num
            sortedArea2num[area][1] = allNum
        num80 = len(self.macros) * 0.8
        num90 = len(self.macros) * 0.9
        flag80 = True
        for area,[_,accNum] in sortedArea2num.items():
            if flag80 and accNum >= num80:
                self.macroArea80 = area
                flag80 = False
                
            if accNum >= num90:
                self.macroArea90 = area
                break
    
    def updateDataFixedMacro(self, _beta_z = 1):
        #########################
        # 更新pin相对左下角的位置 #
        #########################
        for pin in self.pins:
            node = self.nodes[pin.nodeName]
            pin.offsetX = pin.centerOffsetX + node.width / 2 
            pin.offsetY = pin.centerOffsetY + node.height / 2 
            pin.offsetZ = self.layerDepth / 2
        
        ###########
        # 缩小版图 #
        ###########
        
        # 不缩小版图
       
        
        ##################
        # 设定单元z轴深度 #
        ##################
        
        # self.layerDepth = self.rowHeight * layerDepthFactor # 层高等于行高的多倍
        self.layerDepth = 1
        self.z_max = self.layerDepth * self.numLayer 
        
        
        beta_z = _beta_z
        if beta_z == -1:
            beta_z = self.numLayer / self.numRows * 100
        # beta_z = self.numLayer / self.numRows * 100
        Logger.printPlace(f"beta_z: {beta_z}")
        self.z_max = int(max(self.dieWidth,self.dieHeight) * beta_z)
        self.layerDepth = int(self.z_max / self.numLayer)
        
        
        Logger.printPlace(f"zh: {self.z_max} , layerDepth: {self.layerDepth}")
        # self.layerDepth = int((self.dieWidth + self.dieHeight) / (2 * self.numLayer)) # 方案1 单元高度为平均高度
        
        
        ################   
        # 更新node2net #
        ################
        for netName,net in self.nets.items():
            for i in net.pinIndex:
                pin = self.pins[i]
                nodeName = pin.nodeName
                if nodeName in self.node2Net:
                    self.node2Net[nodeName].append(netName)
                else:
                    _list = []
                    _list.append(netName)
                    self.node2Net[nodeName] = _list 

       
        # 直接以macro的key作为目标
        self.macroConnect = list(self.macros.keys())
        
        #########################
        # 统计cell面积与宏块面积 #
        #########################
        macroArea = 0 # 宏块的面积和
        nodeArea = 0 # 所有节点的面积和
        for _, macro in self.macros.items():
            macroArea += macro.width * macro.height
            
        for _, node in self.nodes.items():
            nodeArea += node.width * node.height
        self.macroArea = macroArea
        self.cellArea = nodeArea - macroArea
        self.allArea = self.dieWidth * self.dieHeight * self.numLayer 
        
        ###############################
        # 计算macro的平均宽度和平均高度 #
        ###############################
        widthAll = 0
        heightAll = 0
        for _, macro in self.macros.items():
            widthAll += macro.width
            heightAll += macro.height
        self.averageMacroWidth = int(widthAll / len(self.macros))
        self.averageMacroHeight = int(heightAll / len(self.macros))
        
        #################################
        # 计算宏块面积80% 90%所对应的面积 #
        #################################
        area2num = {} # key=area value = [num, accNum]  面积 数量 累计数量
        for _, macro in self.macros.items():
            area = macro.width * macro.height
            if area in area2num:
                area2num[area][0] += 1
            else:
                area2num[area] = [1, 0] 
        sortedArea2num = {key: area2num[key] for key in sorted(area2num.keys())}
        allNum = 0
        for area,[num,_] in sortedArea2num.items():
            allNum += num
            sortedArea2num[area][1] = allNum
        num80 = len(self.macros) * 0.8
        num90 = len(self.macros) * 0.9
        flag80 = True
        for area,[_,accNum] in sortedArea2num.items():
            if flag80 and accNum >= num80:
                self.macroArea80 = area
                flag80 = False
                
            if accNum >= num90:
                self.macroArea90 = area
                break
         
    def statsInfo(self):
        # print(self.numNets)
        # print(self.numPins)
        # exit(11)
        
        # 统计总面积
        areaAll = 0
        for nodeName, node in self.nodes.items():
            area = node.width * node.height
            areaAll += area
        print(f"areaAll:{areaAll}")
        exit(0)
        # 统计宏块与其相联的模块的数量
        '''
        macroConnect = set()
        for macroName, macro in self.macros.items():
            #遍历每个宏块
            macroConnect.add(macroName)
            netList = self.node2Net[macroName]
            for netName in netList:
                #遍历宏块相连的每条边
                net = self.nets[netName]
                for i in net.pinIndex:
                    #遍历边的每个点
                    pin = self.pins[i]
                    nodeName = pin.nodeName
                    macroConnect.add(nodeName)
        '''
        
        # 统计宏块面积与数量
        '''
        area2num = {}
        for nodeName, node in self.nodes.items():
            if nodeName in self.macros:
                continue
            area = node.width #* node.height
            if area in area2num:
                area2num[area] += 1
            else:
                area2num[area] = 1
        
        sortedArea2num = {key: area2num[key] for key in sorted(area2num.keys())}
        with open("statsInfo/cellWidth/"+self.fileName+".csv","w",encoding="utf-8") as f:
            for area,num in sortedArea2num.items():
                f.write(f"{area},{num}\n")
        '''
        # 统计macro连通性
        #'''
        vertices = self.macroConnect
        hyperedges = []
        for netName, net in self.macroConnectNets.items():
            nodeNameList = []
            for i in net.pinIndex:
                pin = self.pins[i]
                nodeNameList.append(pin.nodeName)
            hyperedges.append(nodeNameList)
        connected_components = find_connected_components(vertices, hyperedges)
        #'''
        # 统计node的连通性
        '''
        vertices = list(self.nodes.keys()) 
        hyperedges = []
        for netName, net in self.nets.items():
            nodeNameList = []
            for i in net.pinIndex:
                pin = self.pins[i]
                nodeNameList.append(pin.nodeName)
            hyperedges.append(nodeNameList)
        connected_components = find_connected_components(vertices, hyperedges)
        '''
        # 输出每个连通子图
        print("######################################")
        print(f"{self.auxFileName} having {len(connected_components)} Connected subgraph, as follows:")
        print("Only output the first 10.")
        for i, component in enumerate(connected_components):
            print(f"connected subgraph {i + 1} : {component[:10]} size {len(component)}")
        print("######################################\n")
        exit(0)
        
    def generateMacroConnectBookShelf(self, outputDir:str, fileName:str, curlayer:int = -1, moveMacro = False,macrolegal = -1, tooManyMacro = False):
        """ 第一个参数表示一级文件夹 temp
            第二个参数表示名字 adaptec1
            则形成路径   outputDir/fileName/layer_1/fileName.xx   
            eg.  bookshelf/adaptec1/layer_1/adaptec1.aux
            第三个参数 curlayer 表示生成的文件所在层数, 只在2d时用得到(mix2d,cell2d)
            第四个参数 moveMacro表示是否可以移动宏块, 在Mixed环节需要用到(mix3d,mix2d)
        """
        # bookshelf/
        if not os.path.isdir(outputDir):  
            Logger.printPlace(outputDir + " folder does not exist, create folder")
            os.system("mkdir -p %s" % (outputDir))
        # bookshelf/adaptec1
        fileNameDir = os.path.join(outputDir, fileName) 
        if not os.path.isdir(fileNameDir):
            Logger.printPlace(fileNameDir + " folder does not exist, create folder")
            os.system("mkdir -p %s" % (fileNameDir))
        
        fileNameSecondDir = fileNameDir # 不分层的话，用两层目录就够了，不需要layer级别
        if curlayer != -1:
            # bookshelf/adaptec1/layer_1
            fileNameSecondDir = os.path.join(fileNameDir, "layer_"+str(curlayer))
            if not os.path.isdir(fileNameSecondDir):
                Logger.printPlace(fileNameSecondDir + " folder does not exist, create folder")
                os.system("mkdir -p %s" % (fileNameSecondDir))
            if macrolegal != -1:
                # bookshelf/adaptec1/layer_1/macrolegal_1
                fileNameSecondDir = os.path.join(fileNameSecondDir, "macrolegal_"+str(macrolegal))
                if not os.path.isdir(fileNameSecondDir):
                    Logger.printPlace(fileNameSecondDir + " folder does not exist, create folder")
                    os.system("mkdir -p %s" % (fileNameSecondDir))
                if tooManyMacro: # 还得多加一层目录
                    # bookshelf/adaptec1/layer_1/macrolegal_1/minMacrolegal
                    fileNameSecondDir = os.path.join(fileNameSecondDir, "minMacrolegal")
                    if not os.path.isdir(fileNameSecondDir):
                        Logger.printPlace(fileNameSecondDir + " folder does not exist, create folder")
                        os.system("mkdir -p %s" % (fileNameSecondDir))
                
        
        auxFile = self.generateAux(fileNameSecondDir, fileName)
        self.generateWts(fileNameSecondDir, fileName)
        self.generateScl(fileNameSecondDir, fileName, tooManyMacro)
        
        # MacroConnect的相关生成函数 只生成macro及其连接的点
        # self.generateMacroConnectNetsOld(fileNameSecondDir, fileName)
        self.generateMacroConnectNets(fileNameSecondDir, fileName)
        
        nodeInThisLayer = self.generateMacroConnectNodes(fileNameSecondDir, fileName, curlayer, moveMacro, tooManyMacro)
        self.generateMacroConnectPl(fileNameSecondDir, fileName, curlayer, moveMacro, tooManyMacro)  
        return auxFile, nodeInThisLayer
    
    def generateMacroConnectNets(self, outputFileDir, fileName):
        # 目前只考虑当前层的net 
        headerStr="UCLA nets 1.0\n"
        headerStr+="# Created  : {}\n".format(datetime.datetime.now().strftime('%Y %m %d'))
        headerStr+="# User     : cjq\n\n"
        numberStr=""
        entriesStr=""

        numAllPin = 0 #记录macro以及其连接的单元组成的net中所有的pin的数量
        nodeNameStr = set()
        for netName, net in self.macroConnectNets.items():
            netPinEntries = ""
            numPin = 0
            for i in net.pinIndex:
                pin = self.pins[i]
                netPinEntries += "  {}  {} :  {:0.6f}  {:0.6f}\n".format(pin.nodeName, "I", pin.centerOffsetX, pin.centerOffsetY)
                numPin += 1
            numAllPin += numPin
            entriesStr += "NetDegree : {}   {}\n".format(numPin, net.netName)
            entriesStr += netPinEntries

        numberStr+="NumNets : {}\n".format(len(self.macroConnectNets))
        numberStr+="NumPins : {}\n\n".format(numAllPin)
        filePath = os.path.join(outputFileDir, fileName+".nets")
        with open(filePath,"w",encoding="utf-8") as f:
            f.write(headerStr)
            f.write(numberStr)
            f.write(entriesStr)
    
    def generateMacroConnectNetsOld(self, outputFileDir, fileName):
        # 目前只考虑当前层的net 
        headerStr="UCLA nets 1.0\n"
        headerStr+="# Created  : {}\n".format(datetime.datetime.now().strftime('%Y %m %d'))
        headerStr+="# User     : cjq\n\n"
        numberStr=""
        entriesStr=""

        netInMacroConnect = [] # 记录由macro以及其连接的单元组成的net
        numAllPin = 0 #记录macro以及其连接的单元组成的net中所有的pin的数量
        nodeNameStr = set()
        for netName, net in self.nets.items():
            numPinInMacroConnect = 0
            nodeNameSet = []
            for i in net.pinIndex:
                # 先遍历一遍判断该net是否连接两个以上的单元
                pin = self.pins[i]
                if pin.nodeName in self.macroConnect and pin.nodeName not in nodeNameSet:
                    # 去除重复的
                    nodeNameSet.append(pin.nodeName)
                    numPinInMacroConnect += 1
            nodeNameSetSort = sorted(nodeNameSet) # 有序
            # 将nodeName变成string
            str = ""
            for nodeName in nodeNameSetSort:
                str += nodeName
            
            if numPinInMacroConnect >= 2 :#and str not in nodeNameStr:
                netInMacroConnect.append(netName)
                nodeNameStr.add(str)
        
        for netName in netInMacroConnect:
            net = self.nets[netName]
            netPinEntries = ""
            numPin = 0
            nodeNameSet = set()
            for i in net.pinIndex:
                # 第二次遍历 拿到net当中的包含在macro的单元
                pin = self.pins[i]
                # if pin.nodeName in self.macroConnect：
                if pin.nodeName in self.macroConnect and pin.nodeName not in nodeNameSet:
                    # 去除重复的
                    nodeNameSet.add(pin.nodeName)
                    netPinEntries += "  {}  {} :  {:0.6f}  {:0.6f}\n".format(pin.nodeName, "I", pin.centerOffsetX, pin.centerOffsetY)
                    numPin += 1
            numAllPin += numPin
            entriesStr += "NetDegree : {}   {}\n".format(numPin, net.netName)
            entriesStr += netPinEntries

        numberStr+="NumNets : {}\n".format(len(netInMacroConnect))
        numberStr+="NumPins : {}\n\n".format(numAllPin)
        filePath = os.path.join(outputFileDir, fileName+".nets")
        with open(filePath,"w",encoding="utf-8") as f:
            f.write(headerStr)
            f.write(numberStr)
            f.write(entriesStr)
    
    def changeMacro(self, width, height,averageMacro = False):
        multiple = 1.2
        if not averageMacro:
            return int(width*multiple), int(height*multiple)
        else:
            return self.averageMacroWidth, self.averageMacroHeight
    
    def generateMacroConnectNodes(self, outputFileDir, fileName, curlayer, moveMacro, tooManyMacro):
        headerStr="UCLA nodes 1.0\n"
        headerStr+="# Created  : {}\n".format(datetime.datetime.now().strftime('%Y %m %d'))
        headerStr+="# User     : cjq\n\n"
        numberStr=""
        entriesStr=""
        terminalStr=""
        terminalNIStr=""
        numTerminal = 0
        numTerminalNI = 0
        numMove = 0
        nodeInThisLayer = False
        
        # 3D情况下
        if curlayer == -1: # 只在3D的时候扩大
            for nodeName in self.macroConnect:
                node = self.nodes[nodeName]
                if node.attribute == None or (moveMacro and nodeName != "v1"):
                    # nodeWidth, nodeHeight = self.changeMacro(node.width, node.height, averageMacro = False)  # 将宏块平均
                    # 超过80%的置为80%的面积
                    nodeWidth = node.width
                    nodeHeight = node.height
                    # 将面积大于80%的宏块变小
                    # area = nodeWidth * nodeHeight
                    # if area > self.macroArea80:
                    #     ratio = math.sqrt(self.macroArea80 / area)
                    #     nodeWidth *= ratio
                    #     nodeHeight *= ratio
                    entriesStr+=" {}  {}  {}\n".format(nodeName, int(nodeWidth*1.2), int(nodeHeight*1.2))
                    # entriesStr+=" {}  {}  {}\n".format(nodeName, int(nodeWidth), int(nodeHeight))
                    numMove += 1
                else:
                    entriesStr+=" {}  {}  {}  {}\n".format(nodeName, node.width, node.height, node.attribute)
                    # entriesStr+=" {}  {}  {}  {}\n".format(nodeName, node.width, node.height, node.attribute)
                    numTerminal += 1
        # 2D情况下
        else:
            for nodeName in self.macroConnect:
                node = self.nodes[nodeName]
                if node.layer == curlayer: 
                    if node.attribute == None or (moveMacro and nodeName != "v1"):
                        # moveMacro 为True则一直输出为None的情况
                        nodeInThisLayer = True
                        entriesStr += " {}  {}  {}\n".format(nodeName, node.width, node.height)
                        numMove += 1
                    else:
                        if not moveMacro and tooManyMacro: 
                            # 将最小的宏块看成可移动的
                            if node.height == self.minMacroRowHeight:
                                nodeInThisLayer = True
                                entriesStr += " {}  {}  {}\n".format(nodeName, node.width, node.height)
                                numMove += 1
                            else:   
                                terminalStr += " {}  {}  {}  {}\n".format(nodeName, node.width, node.height, node.attribute)
                                numTerminal += 1
                        else:   
                            terminalStr += " {}  {}  {}  {}\n".format(nodeName, node.width, node.height, node.attribute)
                            numTerminal += 1
                else:
                    #不在同一层的都修改为terminal_NI
                    terminalNIStr += " {}  {}  {}  terminal_NI\n".format(nodeName, node.width, node.height)
                    numTerminalNI += 1
           
        numberStr+="NumNodes :   {}\n".format(numMove + numTerminal + numTerminalNI)
        numberStr+="NumTerminals :   {}\n\n".format(numTerminal + numTerminalNI)
        filePath = os.path.join(outputFileDir, fileName+".nodes")
        with open(filePath,"w",encoding="utf-8") as f:
            f.write(headerStr)
            f.write(numberStr)
            f.write(entriesStr)
            if terminalStr:
                f.write(terminalStr)
            if terminalNIStr:
                f.write(terminalNIStr)
        return nodeInThisLayer
           
    def generateMacroConnectPl(self, outputFileDir, fileName, curlayer, moveMacro, tooManyMacro):
        headerStr="UCLA pl 1.0\n"
        headerStr+="# Created  : {}\n".format(datetime.datetime.now().strftime('%Y %m %d'))
        headerStr+="# User     : cjq\n\n"
        entriesStr=""
        fixedStr=""
        fixedNIStr=""
        # 3D情况下
        if curlayer == -1:
            for nodeName in self.macroConnect:
                node = self.nodes[nodeName]
                if node.moveType == None or (moveMacro and nodeName != "v1" ):
                    entriesStr+=" {}  {:.4f}  {:.4f} : {}\n".format(nodeName, node.x, node.y, node.direction)
                else:
                    entriesStr+=" {}  {:.4f}  {:.4f} : {} {}\n".format(nodeName, node.x, node.y, node.direction, node.moveType)
        # 2D情况下
        else:
            for nodeName in self.macroConnect:
                node = self.nodes[nodeName]
                if node.layer == curlayer:
                    if node.moveType == None or (moveMacro and nodeName != "v1"):
                        # moveMacro 为真则一直输出为None的情况，让macro可移动
                        entriesStr+=" {}  {}  {} : {}\n".format(nodeName, node.x, node.y, node.direction)
                    else:
                        if not moveMacro and tooManyMacro: 
                            # 将最小的宏块看成可移动的
                            if node.height == self.minMacroRowHeight:
                                entriesStr+=" {}  {}  {} : {}\n".format(nodeName, node.x, node.y, node.direction)
                            else:
                                fixedStr+=" {}  {}  {} : {} {}\n".format(nodeName, node.x, node.y, node.direction, node.moveType)
                        else:
                            fixedStr+=" {}  {}  {} : {} {}\n".format(nodeName, node.x, node.y, node.direction, node.moveType)
                else:
                    #不在同一层的都修改为 /FIXED_NI
                    fixedNIStr+=" {}  {}  {} : {} /FIXED_NI\n".format(nodeName, node.x, node.y, node.direction)
        
        filePath = os.path.join(outputFileDir, fileName+".pl")
        with open(filePath,"w",encoding="utf-8") as f:
            f.write(headerStr)
            f.write(entriesStr)
            if fixedStr:
                f.write(fixedStr)
            if fixedNIStr:
                f.write(fixedNIStr)
            
    def addVirtualMacro(self):
        # 每层添加一个虚拟宏块
        name = "v1"
        # 剩余位置应该能够放下macro
        # spaceArea = self.allArea - int(self.macroArea * 1.5)
        # spaceArea = self.cellArea #* 0.6
        # spaceArea = self.cellArea * 0.8
        # spaceArea = self.cellArea * 1.2
        spaceArea = self.cellArea * 1.4
        layerCellAreaWidth = int(math.sqrt(spaceArea / self.numLayer))
        layerNumRow = int(layerCellAreaWidth / self.rowHeight)
        height = layerNumRow * self.rowHeight
        width = int((spaceArea / self.numLayer) / height)
        Logger.printPlace(f"VirtualMacro v1: width {width} , height {height}")
        x = (self.dieWidth - width) / 2
        y = (self.dieHeight - height) / 2
        virtualMacro = Node(name, width, height, "terminal")
        virtualMacro.x = x
        virtualMacro.y = y
        virtualMacro.layer = 0
        virtualMacro.moveType = "/FIXED"
        # 添加到 nodes macros 当中
        self.nodes[name] = virtualMacro
        self.macros[name] = virtualMacro
        self.macroConnect.append(name)
        
        # 添加net 每个macro都与虚拟macro相连
        # self.addVirtualNet()
    
    def addVirtualNet(self):
        # 添加net 每个macro都与虚拟macro相连
        numVirtualPins = len(self.macros)
        netName = "vn1"
        net = Net(netName, len(self.pins), numVirtualPins)
        self.nets[netName] = net
        for macroName, macro in self.macros.items():
            nodeName = macroName
            centerX = 0
            centerY = 0
            # pin均放在中心点
            pin = Pin(nodeName, centerX, centerY)
            self.pins.append(pin)     
        
    def removeNode(self):
        nodeName = "v1"
        netName = "vn1"
        if nodeName in self.nodes:
            del self.nodes[nodeName]
        if nodeName in self.macros:
            del self.macros[nodeName]
        if netName in self.nets:
            del self.nets[netName]
       
    def drawMacroConnect(self):
        # 给定的节点列表和超边
        nodes = self.macroConnect
        hyperedges = []
        for _, net in self.nets.items():
            nodeSet = set()
            for i in net.pinIndex:
                pin = self.pins[i]
                if pin.nodeName in nodes:
                    nodeSet.add(pin.nodeName)
            if len(nodeSet) > 1: # 两个点以上的边才认为是边
                hyperedges.append(nodeSet)
                
        # 创建一个有向图
        G = nx.DiGraph()

        # 添加节点和超边
        G.add_nodes_from(nodes)
        for hyperedge in hyperedges:
            hyperedge = list(hyperedge)
            source = hyperedge[0]
            targets = hyperedge[1:]
            for target in targets:
                G.add_edge(source, target)

        # 设置画布大小
        sizeX = 10
        sizeY = 5
        plt.figure(figsize=(sizeX, sizeY*self.numLayer)) # 不同层则 y+sizeY
        
        # 给每个节点添加一个类别属性（根据层数）
        node_attributes = {}
        pos = {}
        for nodeName in nodes:
            node = self.nodes[nodeName]
            node_attributes[nodeName] = node.layer
            x = round(node.x * sizeX / self.dieWidth, 2)
            y = round(node.y * sizeY / self.dieWidth, 2) + node.layer * sizeY
            pos[nodeName] = [x, y]
            
        nx.set_node_attributes(G, node_attributes, 'category')

        # 定义类别到颜色的映射
        category_colors = {
            0: 'skyblue',
            1: 'lightgreen',
            2: 'salmon',
            3: 'purple'
        }

        # 根据节点的类别属性分配颜色
        node_colors = [category_colors[G.nodes[node]['category']] for node in G.nodes]

        # 检查是否是DAG
        if nx.is_directed_acyclic_graph(G):
            print("Graph is a DAG")
        else:
            print("Graph is not a DAG")
        
        

        # 可视化DAG并根据类别着色
        # pos = nx.spring_layout(G)  # 使用spring layout进行布局
        # pos自定义布局
        nx.draw(G, pos, with_labels=True, arrows=True, node_size=700, node_color=node_colors, font_size=5, font_color="black", edge_color="gray")
        
        fileName  = f"statsInfo/macroVisualization/{self.fileName}.png"
        print(f"generate fig: {fileName}")
        plt.title(f"{self.fileName} macro connect figure")
        
        plt.savefig(fileName, format="png", bbox_inches="tight")
        
        exit(1)
 
    def statsMacro(self):
        
        standardNum = len(self.nodes) - len(self.macros)
        
        allArea = 0
        for _,node in self.nodes.items():
            allArea += node.width * node.height
        allAverageArea = int(allArea / len(self.nodes))
        times = lambdaTimes[self.auxFileName]
        limitArea = allAverageArea * times 
        print(f"limitArea:{limitArea}")
        # 超过 limitArea 才能被认作是Macro，否则为 I/O,设置为可移动且面积为0
        # macroNum = 0
        # IONum = 0
        # for macroName, macro in self.macros.items():
        #     area = macro.width * macro.height
        #     if area > limitArea:
        #         macroNum += 1
        #     else:
        #         IONum += 1
        # print(f"{macroNum}, {IONum}")
        
        # 找到孤立的宏块

        # for netName, net in self.nets.items():
        #     if netName == "n5869" or netName == "n5870":
        #         print(netName)
        #         for i in net.pinIndex:
        #             pin = self.pins[i]
        #             node = self.nodes[pin.nodeName]
        #             area = node.width * node.height
        #             print(f"nodeName:{pin.nodeName} area:{area}", )
        
        # stats macro num
        besideRegion = set() # 需要除掉的单元 用于后续清理
        macroNum = 0
        cell2MacroNum = 0  # 只针对bigblue3
        for nodeName, node in self.macros.items():
            area = node.width * node.height
            if area > limitArea :
                macroNum += 1
            else:
                besideRegion.add(nodeName)
        IONum = len(self.macros) - macroNum
        
        if self.auxFileName == "bigblue3":
            for nodeName, node in self.nodes.items():    
                if nodeName not in self.macros:
                    area = node.width * node.height
                    if area == 2856: # 设置成macro
                        cell2MacroNum += 1
                        node.attribute = "terminal"
                        node.moveType = "/FIXED"
                        self.macros[nodeName] = node
        
        print(f"obj:{len(self.nodes)}, Movable obj:{len(self.nodes) - IONum}, standard Cell:{standardNum- cell2MacroNum},Macro:{macroNum + cell2MacroNum},I/O:{IONum}")
        # 统计最大cells面积
        # areaSet = set()
        # cellsName = []
        # for nodeName, node in self.nodes.items():
        #     if nodeName not in self.macros:
        #         cellsName.append(nodeName)
        #         area = node.width * node.height
        #         areaSet.add(area)
        # areaSorted = sorted(list(areaSet))
        
        # num = 0
        # maxArea = areaSorted[-1]  # = 2856
        # for nodeName in cellsName:
        #     node = self.nodes[nodeName]
        #     area = node.width * node.height
        #     if area >= maxArea:
        #         num += 1
        # print(areaSorted)
        # print(f"maxArea:{maxArea}")
        # print(f"standard Cell:{standardNum - num},Macro:{macroNum + num}")
        
        ####################
        # 去掉属于I/O的宏块 #
        ####################
            
        # 先清理net
        besideNet = [] #需要删除的边
        for netName, net in self.nets.items():
            besidePin = []
            #先遍历
            for i in net.pinIndex:
                pin = self.pins[i]
                if pin.nodeName in besideRegion:
                    besidePin.append(i)
            #再删除
            if len(net.pinIndex) - len(besidePin) > 1 : #所剩下的点要大于1才作为边  !!! 原本的数据集中是存在一个net中只存在一个pin的情况的
                for i in besidePin:
                    net.pinIndex.remove(i)
            else:
                #需要整个边进行删除
                besideNet.append(netName)
        for netName in besideNet:
            del self.nets[netName]
        # 删除nodes macros  
        # pins不用删除，不影响generateBookshelf
        for macroName in besideRegion:
            del self.nodes[macroName]
            del self.macros[macroName]
        # 更新数量
        self.numNodes = len(self.nodes)
        self.numTerminals = len(self.macros)
        self.numNets = len(self.nets)
        numPins = 0
        for _, net in self.nets.items():
            numPins += len(net.pinIndex)
        self.numPins = numPins
   
    def countNet(self):
        macro_macro = 0
        macro_cell = 0
        cell_cell = 0
        for _, net in self.nets.items():
            hasMacro = False
            hasCell = False
            for i in net.pinIndex:
                pin = self.pins[i]
                node = self.nodes[pin.nodeName]
                if node.attribute == "terminal":
                    #宏块
                    hasMacro = True
                else:
                    hasCell = True
            if hasMacro and hasCell:
                macro_cell += 1
            elif not hasMacro and hasCell:
                cell_cell += 1
            elif hasMacro and not hasCell:    
                macro_macro += 1
        return macro_macro, macro_cell, cell_cell
        
    def generateLLMData(self):
        widthAver = 0
        heightAver = 0
        
        for macroName,macro in self.macros.items():
            widthAver += macro.width
            heightAver += macro.height
        widthAver /= len(self.macros)
        heightAver /= len(self.macros)
        xh = int(self.dieWidth/widthAver) - 1
        yh = int(self.dieHeight/heightAver) - 1
        
        netStr = ""
        ii = 0
        for netName,net in self.macroConnectNets.items():
            netStr += f"net{ii} "
            for i in net.pinIndex:
                pin = self.pins[i]
                netStr += f"{pin.nodeName} "
            netStr += "\n"
            ii += 1
        
        with open("LLMData/"+self.fileName+".txt","w",encoding="utf-8") as f:
            f.write(f"layer {self.numLayer}\n")
            f.write(f"size [0,{xh}] [0,{yh}]\n")   
            f.write(netStr)
  
    def updateChipSizeAndNumLayer(self, qwen_pre):
        dieWidth, dieHeight, numLayer, zh = qwen_pre[self.auxFileName]
        self.dieWidth = dieWidth
        self.dieHeight = dieHeight
        self.numLayer = numLayer
        self.z_max = zh
        self.layerDepth = int(self.z_max / self.numLayer)
        Logger.printPlace(f"new dieWidth: {self.dieWidth},dieHeight: {self.dieHeight}, numLayer: {self.numLayer}")
        Logger.printPlace(f"new zh: {self.z_max} , layerDepth: {self.layerDepth}")
         
    def stats2layer(self):
        kahyparLayerFile = "benchmarks/kahyparLayerFileISPD2006/newblue7_2_layer.txt"
        macroLayerFile = "temp_new0416_zAlpha0.25/a_mix3d/result/newblue7_2/newblue7_2.layer.txt"
        cellLayerFile = "temp_new0416_zAlpha0.25/c_cell3d/result/newblue7_2/newblue7_2.layer.txt"
        
        macroNums = [0, 0]
        macroArea = [0, 0]
        cellNums = [0, 0]
        cellArea = [0, 0]
        with open(kahyparLayerFile) as inFile:
            lines = inFile.readlines()
            for line in lines:
                words = line.strip().split()
                nodeName = words[0]
                layer = int(words[1])
                node = self.nodes[nodeName]
                area = node.width * node.height
                #判断是否是宏块
                if nodeName in self.macros:
                    macroNums[layer] += 1;
                    macroArea[layer] += area
                else:
                    cellNums[layer] += 1;
                    cellArea[layer] += area
        print(f"kahyparLayerFile,macro,{macroNums[0]},{macroNums[1]},{macroArea[0]},{macroArea[1]}\
                                 ,cell,{cellNums[0]},{cellNums[1]},{cellArea[0]},{cellArea[1]}")
        
        macroNums = [0, 0]
        macroArea = [0, 0]
        cellNums = [0, 0]
        cellArea = [0, 0]
        with open(macroLayerFile) as inFile:    
            lines = inFile.readlines()
            for line in lines:
                words = line.strip().split()
                nodeName = words[0]
                layer = int(words[1])
                node = self.nodes[nodeName]
                area = node.width * node.height
                #判断是否是宏块
                if nodeName in self.macros:
                    macroNums[layer] += 1;
                    macroArea[layer] += area
                else:
                    cellNums[layer] += 1;
                    cellArea[layer] += area
        print(f"macroLayerFile,macro,{macroNums[0]},{macroNums[1]},{macroArea[0]},{macroArea[1]}\
                                 ,cell,{cellNums[0]},{cellNums[1]},{cellArea[0]},{cellArea[1]}")
        
        macroNums = [0, 0]
        macroArea = [0, 0]
        cellNums = [0, 0]
        cellArea = [0, 0]
        with open(cellLayerFile) as inFile:   
            lines = inFile.readlines()
            for line in lines:
                words = line.strip().split()
                nodeName = words[0]
                layer = int(words[1])
                node = self.nodes[nodeName]
                area = node.width * node.height
                #判断是否是宏块
                if nodeName in self.macros:
                    macroNums[layer] += 1;
                    macroArea[layer] += area
                else:
                    cellNums[layer] += 1;
                    cellArea[layer] += area
        print(f"cellLayerFile,macro,{macroNums[0]},{macroNums[1]},{macroArea[0]},{macroArea[1]}\
                                 ,cell,{cellNums[0]},{cellNums[1]},{cellArea[0]},{cellArea[1]}")
        exit()
        
    def readpyG(self, x, cond):
        # 缩放比例， [-1, 1] * [-1 , 1] 缩放到rowheight为16
        rowNum = cond.numRow
        rowheight = 2 / rowNum
        ROWHEIGHTTRUE = 16
        scale = ROWHEIGHTTRUE / rowheight
        
        # die的尺寸，用于生成 scl
        self.dieWidth = round(2 *scale)
        self.dieHeight = rowNum * ROWHEIGHTTRUE
        self.numRows = rowNum
        self.rowHeight = ROWHEIGHTTRUE
        # node  pl
        numNodes = cond.x.shape[0]
        for id in range(numNodes):
            nodeName = f"a{id}"
            x_size = round(cond.x[id][0].item() * scale)
            y_size = round(cond.x[id][1].item() * scale)
            node = Node(nodeName, x_size, y_size, "terminal" if cond.is_macros[id] or cond.is_ports[id] else None)
            xx = (x[id][0] + 1) * scale  # +1是为了将坐标轴变成左下角
            yy = (x[id][1] + 1) * scale
            node.update(xx, yy, "N", "/FIXED" if cond.is_macros[id] or cond.is_ports[id] else None)
            self.nodes[nodeName] = node
            if cond.is_macros[id] or cond.is_ports[id]:
                self.macros[id] = node
        # net  目前的net都是无向边
        if False: #cond.edge_pin_id:
            netData = cond.edge_index.T
            pinData = cond.edge_attr
            edge2net = cond.edge_pin_id

            # 按 netId 分组 edge
            net_groups = defaultdict(list)
            for e, netId in enumerate(edge2net):
                net_groups[netId].append(e)

            # 构造每个 Net
            for netId, edge_indices in net_groups.items():
                netName = f"net{netId}"
                netSize = len(edge_indices) * 2  # 每条 edge 提供两个 pin
                net = Net(netName, len(self.pins), netSize)
                self.nets[netName] = net

                for e in edge_indices:
                    src, dst = netData[e]
                    dx1, dy1, dx2, dy2 = pinData[e]

                    nodeName1 = f"a{src}"
                    nodeName2 = f"a{dst}"

                    pin1 = Pin(nodeName1, float(dx1) * scale, float(dy1) * scale)
                    pin2 = Pin(nodeName2, float(dx2) * scale, float(dy2) * scale)

                    self.pins.append(pin1)
                    self.pins.append(pin2)
        else:
            numNets = cond.edge_index.shape[1] // 2
            netSize = 2  # 目前所有边都是普通边
            netData = cond.edge_index.T
            pinData = cond.edge_attr  # edge_attr[e] = [src_dx, src_dy, sink_dx, sink_dy]
            for netId in range(numNets):
                netName = f"net{netId}"
                net = Net(netName, len(self.pins), netSize)
                self.nets[netName] = net
                # 两个pin
                nodeName1 = f"a{netData[netId][0]}"
                nodeName2 = f"a{netData[netId][1]}"
                pin1centerX = float(pinData[netId][0]) * scale
                pin1centerY = float(pinData[netId][1]) * scale
                pin2centerX = float(pinData[netId][2]) * scale
                pin2centerY = float(pinData[netId][3]) * scale

                pin1 = Pin(nodeName1, pin1centerX, pin1centerY)
                pin2 = Pin(nodeName2, pin2centerX, pin2centerY)
                self.pins.append(pin1)
                self.pins.append(pin2)


        
    
            
if __name__ == "__main__":

    database = DataBase()
    database.readBookshelf("../benchmarks/ispd2005/adaptec2/adaptec2.aux")
    database.parsePLFile("../eDensity3dResult/adaptec1/adaptec1.gp.pl")