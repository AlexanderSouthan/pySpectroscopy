# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (QMainWindow, QComboBox, QWidget, QGridLayout,
                             QDesktopWidget, QLabel, QVBoxLayout,
                             QPushButton, QHBoxLayout)


class raman_pca_window(QMainWindow):
    def __init__(self, raman_data):
        self.update_data(raman_data)
        super().__init__()
        self.init_window()
        self.define_widgets()
        self.position_widgets()
        self.connect_event_handlers()

    def init_window(self):
        self.setGeometry(500, 500, 600, 360) #xPos,yPos,width, heigth
        self.center() #center function is defined below
        self.setWindowTitle('Principal component analysis options')

        self.container0 = QWidget(self)
        self.setCentralWidget(self.container0)

        self.grid_container = QGridLayout()
        self.container0.setLayout(self.grid_container)

    def define_widgets(self):
        self.pca_label = QLabel('<b>Principal component analysis</b>')
        self.pca_components_label = QLabel('Number of principal components')
        self.pca_components_combo = QComboBox()
        self.pca_components_combo.addItems([str(ii) for ii in range(1, 13)])
        self.perform_pca_button = QPushButton('Perform PCA')

    def position_widgets(self):
        self.pca_components_layout = QHBoxLayout()
        self.pca_components_layout.addWidget(self.pca_components_label)
        self.pca_components_layout.addWidget(self.pca_components_combo)

        self.imaging_layout = QVBoxLayout()
        self.imaging_layout.addWidget(self.pca_label)
        self.imaging_layout.addLayout(self.pca_components_layout)
        self.imaging_layout.addWidget(self.perform_pca_button)
        self.imaging_layout.addStretch(1)

        self.grid_container.addLayout(self.imaging_layout, *(0, 0), 1, 1)

    def connect_event_handlers(self):
        self.perform_pca_button.clicked.connect(self.perform_pca)

    def update_data(self, raman_data):
        self.raman_data = raman_data

    def center(self):#centers object on screen
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def perform_pca(self):
        pca_components = int(self.pca_components_combo.currentText())
        self.raman_data.principal_component_analysis(pca_components)

        print('PCA finished!')
