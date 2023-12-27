import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, \
    QStackedLayout
from PyQt5.QtGui import QFont, QPixmap, QPainter, QBrush, QPalette
from PyQt5.QtCore import Qt
import cv2
# 引入自己写的模块
from pic_process import Processor


# 用于展示gui最上面的背景
class BackgroundWidget(QWidget):
    def __init__(self):
        super().__init__()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QBrush(Qt.white))
        painter.drawRect(self.rect())
        # 加载背景图片
        background_image = QPixmap("./bk/bk2.jpg")
        # 缩放背景图片以适应窗口大小
        scaled_image = background_image.scaled(self.size(), Qt.IgnoreAspectRatio)
        # 绘制背景图片
        painter.drawPixmap(self.rect(), scaled_image)


# 创建窗口的类
class ImageProcessingWindow(QWidget):
    def __init__(self):
        super().__init__()
        # 设置窗口一些组件
        self.pic = Processor()

        self.setWindowTitle("YOLO自动标注工具")
        self.setGeometry(100, 100, 700, 840)

        self.input_folder_edit = QLineEdit()
        self.input_folder_edit.setText("./images")
        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setText("./labels/")
        self.shape_recognition_button = QPushButton("形状识别")
        self.color_recognition_button = QPushButton("颜色识别")
        # self.image_output_label = QLabel("第三组\n"
        #                                  "组长：陈伟\n"
        #                                  "组员：王毅、温顺发、刘洋浩")
        self.image_output_label = QLabel()

        # 添加一些布局器，布局窗口
        main_layout = QVBoxLayout()
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("输入文件夹路径:"))
        input_layout.addWidget(self.input_folder_edit)
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("输出文件夹路径:"))
        output_layout.addWidget(self.output_folder_edit)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.shape_recognition_button)
        button_layout.addWidget(self.color_recognition_button)

        main_layout.addLayout(input_layout)
        main_layout.addLayout(output_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.image_output_label)

        # 创建垂直布局
        vertical_layout = QVBoxLayout()
        background_widget = BackgroundWidget()
        vertical_layout.addWidget(background_widget)
        vertical_layout.addLayout(main_layout)

        self.setLayout(vertical_layout)
        # 绑定点击触发事件的槽函数
        self.shape_recognition_button.clicked.connect(self.shape_recognition)
        self.color_recognition_button.clicked.connect(self.color_recognition)

        # 设置样式表
        self.setStyleSheet("""
            QLabel {
                font-size: 18px;
                color: #333333;
            }
            QLineEdit {
                height: 30px;
                font-size: 14px;
                padding: 5px;
                border: 1px solid #cccccc;
            }
            QPushButton {
                height: 30px;
                font-size: 14px;
                background-color: #4caf50;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        # 设置图片输出框显示图片
        pixmap = QPixmap("./bk/bk_d_2.jpg")
        self.image_output_label.setPixmap(pixmap)
        self.image_output_label.setScaledContents(True)

    # 两个槽函数
    def shape_recognition(self):
        # 形状识别槽函数的实现
        dir_path = self.input_folder_edit.text()
        # output_folder_path = self.output_folder_edit.text()
        # 形状识别
        self.pic.shape_match(dir_path)

    def color_recognition(self):
        # 颜色识别槽函数的实现
        dir_path = self.input_folder_edit.text()
        print(dir_path)
        # output_folder_path = self.output_folder_edit.text()
        # 颜色识别
        self.pic.color_match(dir_path)


if __name__ == "__main__":
    # dir_path = "./images"
    # pic = Processor()
    # pic.color_match(dir_path)
    # pic.shape_match(dir_path)
    app = QApplication(sys.argv)
    window = ImageProcessingWindow()
    window.show()
    sys.exit(app.exec_())
