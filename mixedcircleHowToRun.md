## 1、 chipdiffusion宏布局
主目录 /home/pc/data/cjq/work/chipdiffusion-main
1. 运行chipdiffusion得到结果(eval.sh) 在logs/diffusion_debug/ibm.cluster512.v1.eval_guided.300/xxxx
2. 将 pickle 结果转换成ibm的deflef文件：修改实验配置 exp_configs，在主目录运行 python parsing/createUnclusterDEF.py
3. 进入目录benchmarks，将 ibmBookshelfModifyFixedMacro 复制一份作为新数据的容器 如 cp -r ibmBookshelfModifyFixedMacroBest ibmBookshelfModifyFixedMacroBestTime1000
4. 修改replacePlxy.py中的实验配置。运行 python replacePlxy.py。 
5. 最终ibmBookshelfModifyFixedMacroBestTime1000为可以让dreamplace运行的具有初始宏布局的电路数据

## 2、 dreamPlace 标准单元布局
主目录 /home/pc/data/cjq/DREAMPlace-master
1. 进入到benchmarks，将上述文件复制到当前目录 cp -r /home/pc/data/cjq/work/chipdiffusion-main/benchmarks/ibmBookshelfModifyFixedMacroBaselineTime1000 .
2. 生成dreamplace运行的json配置。修改 generateJson.py中的配置，运行python generateJson.py 
3. 进入install目录，修改run.sh，运行./run.sh
4. 修改 getlogs.py ，指定logs，运行 python getlogs.py