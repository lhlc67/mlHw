import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

import Ui_untitled

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow =Ui_untitled.MyMainForm()
    MainWindow.show()
    sys.exit(app.exec_())

