import ast
import astunparse
import astor
import sys
from functionObject import FunctionObject

class Analyzer(ast.NodeTransformer):
    def __init__(self):
        self.objects = []
        pass

    def visit_FunctionDef(self, node):
        """ Visit FunctionDef nodes """
        self.handleVisitFunction(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        """ Visit AsyncFunctionDef nodes """
        self.handleVisitFunction(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        """ Visit ClassDef nodes """
        isFlaky = self.isNodeFlaky(node)
        for functionNode in node.body:
            if isinstance(functionNode, ast.FunctionDef) or isinstance(functionNode, ast.AsyncFunctionDef):
                self.addTestToList(functionNode.name, node.name, isFlaky, functionNode)
        self.generic_visit(node)

    def handleVisitFunction(self, node):
        """ Extender to AsyncFunctionDef and FunctionDef -> Code is similar for both"""
        isFlaky = self.isNodeFlaky(node)
        self.addTestToList(node.name, self.getClass(node), isFlaky, node)

    def addTestToList(self, functionName, className, isFlaky, node):
        """ Create a test object, Add it to the main list """
        # Create Function object
        currentObject = FunctionObject()
        currentObject.setFunctionName(functionName)
        currentObject.setClassName(className)
        currentObject.setBody(self.getBody(node))

        # If it is a TEST
        if self.isTestFunction(node):
            # We mark is as a TEST
            currentObject.setIsTest(True)
            # We mark it as @flaky or not
            if isFlaky:
                currentObject.setIsMarkedFlaky(True)
            else:
                currentObject.setIsMarkedFlaky(False)
        # Else it is a FUNCTION
        else:
            # We mark it as a FUNCTION
            currentObject.setIsTest(False)

        # Then we add it to the test list
        if not self.alreadyInList(currentObject):
            self.objects.append(currentObject)

    def getClass(self, node):
        """ If node is a functionm, get name of the Class """
        if isinstance(node.parent, ast.ClassDef):
            return node.parent.name
        else:
            return None

    def alreadyInList(self, currentObject):
        """ Check if test object is already in list """
        isInList = False
        for element in self.objects:
            if currentObject.__eq__(element):
                isInList = True
                # Handle case if element is already in but MarkedFlaky is False
                if element.getIsMarkedFlaky() == False and currentObject.getIsMarkedFlaky() == True:
                    element.setIsMarkedFlaky(True)
        return isInList

    def isNodeFlaky(self, node):
        """ Check if node is marked with @flaky """
        isFlaky = False
        # If node has annotations
        if len(node.decorator_list) > 0:
            # For each annotation
            for decorator in node.decorator_list:
                # When @flaky has parameters
                if  hasattr(decorator, "func") and hasattr(decorator.func, "id"):
                    if "flaky" in decorator.func.id:
                        isFlaky = True
                # When @flaky has no parameter
                elif hasattr(decorator, "id"):
                    if "flaky" in decorator.id:
                        isFlaky = True
        return isFlaky

    def isTestFunction(self, node):
        """ Check if function is test or CUT """
        isTest = False
        # If node has annotations
        if node.name.startswith("test_"):
            isTest = True
        return isTest

    def getBody(self, node):
        """ Get body of the function """
        # Init body
        body = ""
        # Add body of the function
        for el in node.body:
            body += astor.to_source(el)
        # Add annotation as well
        for el in node.decorator_list:
            annotation = astor.to_source(el).rstrip()
            if "flaky" not in annotation:
                body += annotation
        return body