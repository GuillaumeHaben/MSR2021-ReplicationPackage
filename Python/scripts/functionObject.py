class FunctionObject():
    functionName = ""
    className = ""
    fileName = ""
    projectName = ""
    isMarkedFlaky = False
    isTest = False
    body = ""

    def __init__(self):
        pass

    def getFunctionName(self):
        return self.functionName

    def setFunctionName(self, functionName):
        self.functionName = functionName

    def getClassName(self):
        return self.className

    def setClassName(self, className):
        self.className = className

    def getFileName(self):
        return self.fileName

    def setFileName(self, fileName):
        self.fileName = fileName

    def getProjectName(self):
        return self.projectName

    def setProjectName(self, projectName):
        self.projectName = projectName

    def getBody(self):
        return self.projectName

    def setBody(self, body):
        self.body = body

    def getIsMarkedFlaky(self):
        return self.isMarkedFlaky

    def setIsMarkedFlaky(self, status):
        self.isMarkedFlaky = status

    def getIsTest(self):
        return self.isTest

    def setIsTest(self, status):
        self.isTest = status

    def __eq__(self, other):
        if self.functionName == other.functionName and self.className == other.className:
            return True
        else:
            return False

    def __hash__(self):
        return hash(str(self.functionName) + str(self.className))

    def toString(self):
        print("Function object:")
        print("  functionName:", self.functionName)
        print("  className:", self.className)
        print("  fileName:", self.fileName)
        print("  projectName:", self.projectName)
        print("  isMarkedFlaky:", self.isMarkedFlaky)
        print("  isTest:", self.isTest)
        print("  body:", self.body)

    def toMinString(self):
        print(str(self.projectName) + str(self.fileName) + "::" + str(self.className) + "::" + str(self.functionName))

    def toJSON(self):
        json = {
            "functionName": self.functionName,
            "className": self.className,
            "fileName": self.fileName,
            "projectName": self.projectName,
            "Label": self.isMarkedFlaky,
            "isTest": self.isTest,
            "Body": self.body
        }
        return json