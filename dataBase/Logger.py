
loggingName = 'Place3D'

outputFlag = True
    
def printPlace(messsge:str):
    """格式化输出,只接受一个str参数
    """
    if outputFlag:
        print('[PLACE  ] {} - {}'.format(loggingName, messsge), flush=True)

def printError(messsge:str):
    """格式化输出,只接受一个str参数
    """
    if outputFlag:
        print('[{}  ] {} - {}'.format("ERROR",loggingName, messsge), flush=True)