"""
<name>Kernel Builder</name>
<description>Builds a kernel function.</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/Cost.svg</icon>
<priority>1010</priority>
"""

from Goldenberry.widgets import *
from Goldenberry.widgets.GbDynamicFunctionWidget import GbDynamicFunctionWidget

class GbKernelBuilderWidget(GbDynamicFunctionWidget):
    """Provides a kernel function builder."""
    
    def __init__(self, parent=None, signalManager=None):
        super(GbKernelBuilderWidget, self).__init__(Kernels, GbKernel, parent, signalManager, 'Kernel Function', "Kernel Function")
        
        self.setup_interfaces()
        self.setup_ui()

if __name__=="__main__":
    test_widget()

def test_widget():
    app = QApplication(sys.argv)
    w = GbKernelBuilderWidget()
    w.show()
    app.exec_()


