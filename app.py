import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from src import config
from src.predict import predict_image 

# config dosyasından model yolu ve başlık gibi bilgileri alalım
DEFAULT_MODEL_PATH = config.MODEL_PATH_FOR_APP
APP_TITLE = config.APP_TITLE
IMAGE_SIZE_CONFIG = config.IMAGE_SIZE 

class PredictionThread(QThread):
    prediction_done = pyqtSignal(str, float)
    error_occurred = pyqtSignal(str)

    def __init__(self, image_path, model_path):
        super().__init__()
        self.image_path = image_path
        self.model_path = model_path

    def run(self):
        try:
            predicted_class, confidence = predict_image(self.image_path, model_path=self.model_path)
            if predicted_class is not None and confidence is not None:
                self.prediction_done.emit(predicted_class, confidence)
            else:
                self.error_occurred.emit("Model tahmin yapamadı. Lütfen modelin ve sınıf eşleme dosyasının doğru olduğundan emin olun.")
        except Exception as e:
            self.error_occurred.emit(f"Tahmin sırasında bir hata oluştu: {str(e)}")

class MultiZooApp(QWidget):
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.model_path = DEFAULT_MODEL_PATH 
        self.initUI()

    def initUI(self):
        self.setWindowTitle(APP_TITLE)
        self.setGeometry(100, 100, 700, 500) 

        
        main_layout = QVBoxLayout()

        
        model_layout = QHBoxLayout()
        self.model_path_label = QLabel(f"Kullanılan Model: {os.path.basename(self.model_path)}")
        
        model_layout.addWidget(self.model_path_label)
        
        main_layout.addLayout(model_layout)

       
        self.btn_load_image = QPushButton("Test Görüntüsü Yükle")
        self.btn_load_image.clicked.connect(self.load_image)
        main_layout.addWidget(self.btn_load_image)

       
        self.image_preview_label = QLabel("Buraya bir görüntü yüklendiğinde önizlemesi görünecektir.")
        self.image_preview_label.setAlignment(Qt.AlignCenter)
        self.image_preview_label.setMinimumHeight(250) 
        self.image_preview_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        main_layout.addWidget(self.image_preview_label)

       

        
        self.result_label = QLabel("Tahmin Sonucu:")
        self.result_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.result_label)

        self.confidence_label = QLabel("Güven Skoru:")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.confidence_label)
        
        self.status_label = QLabel("Durum: Hazır") 
        main_layout.addWidget(self.status_label)

        self.setLayout(main_layout)
        self.check_model_exists()

    def check_model_exists(self):
        if not os.path.exists(self.model_path) or not os.path.exists(os.path.join(config.MODEL_SAVE_DIR, config.CLASS_MAPPING_FILE)):
            QMessageBox.warning(self, "Model Hatası", 
                                f"Varsayılan model ({os.path.basename(self.model_path)}) veya sınıf eşleme dosyası bulunamadı.\n"
                                f"Lütfen 'models' klasöründe '{config.BEST_MODEL_NAME}' ve '{config.CLASS_MAPPING_FILE}' dosyalarının bulunduğundan emin olun.\n"
                                "Program bu dosyalar olmadan düzgün çalışmayabilir.")
            self.btn_load_image.setEnabled(False)
            self.status_label.setText("Durum: Model veya sınıf eşlemesi eksik!")
        else:
            self.status_label.setText(f"Durum: {os.path.basename(self.model_path)} ile hazır.")

    def select_model(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Model Seç", config.MODEL_SAVE_DIR, "PyTorch Model Dosyaları (*.pth *.pt)", options=options)
        if fileName:
            
            self.model_path = fileName
            self.model_path_label.setText(f"Kullanılan Model: {os.path.basename(self.model_path)}")
            self.status_label.setText(f"Durum: {os.path.basename(self.model_path)} ile hazır.")
            self.check_model_exists() 
            self.btn_load_image.setEnabled(True) 

    def load_image(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Görüntü Seç", "", "Resim Dosyaları (*.png *.jpg *.jpeg *.bmp)", options=options)
        if fileName:
            self.current_image_path = fileName
            pixmap = QPixmap(fileName)
            
            scaled_pixmap = pixmap.scaled(self.image_preview_label.width(), self.image_preview_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_preview_label.setPixmap(scaled_pixmap)
            
            self.result_label.setText("Tahmin Sonucu:")
            self.confidence_label.setText("Güven Skoru:")
            self.status_label.setText("Durum: Görüntü yüklendi, tahmin ediliyor...")
            self.start_prediction() 
    def start_prediction(self):
        if self.current_image_path and self.model_path:
            if not os.path.exists(self.model_path):
                QMessageBox.critical(self, "Hata", f"Model dosyası bulunamadı: {self.model_path}")
                self.status_label.setText("Durum: Model dosyası eksik!")
                return
            class_mapping_file = os.path.join(os.path.dirname(self.model_path), config.CLASS_MAPPING_FILE)
            if not os.path.exists(class_mapping_file):
                
                class_mapping_file = os.path.join(config.MODEL_SAVE_DIR, config.CLASS_MAPPING_FILE)
                if not os.path.exists(class_mapping_file):
                    QMessageBox.critical(self, "Hata", f"Sınıf eşleme dosyası ({config.CLASS_MAPPING_FILE}) bulunamadı.")
                    self.status_label.setText("Durum: Sınıf eşleme dosyası eksik!")
                    return

            self.btn_load_image.setEnabled(False) 
            
            
            self.prediction_thread = PredictionThread(self.current_image_path, self.model_path)
            self.prediction_thread.prediction_done.connect(self.on_prediction_done)
            self.prediction_thread.error_occurred.connect(self.on_prediction_error)
            self.prediction_thread.start()
        else:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir görüntü yükleyin ve modelin seçili olduğundan emin olun.")
            self.status_label.setText("Durum: Görüntü veya model eksik.")

    def on_prediction_done(self, predicted_class, confidence):
        self.result_label.setText(f"Tahmin Edilen Hayvan Türü: <b>{predicted_class}</b>")
        self.confidence_label.setText(f"Güven Skoru: <b>{confidence:.2%}</b>")
        self.btn_load_image.setEnabled(True)
        
        self.status_label.setText("Durum: Tahmin tamamlandı.")

    def on_prediction_error(self, error_message):
        QMessageBox.critical(self, "Tahmin Hatası", error_message)
        self.result_label.setText("Tahmin Sonucu: Hata")
        self.confidence_label.setText("Güven Skoru: - ")
        self.btn_load_image.setEnabled(True)
        
        self.status_label.setText("Durum: Tahmin sırasında hata.")

    def resizeEvent(self, event):
       
        if self.current_image_path:
            pixmap = QPixmap(self.current_image_path)
            scaled_pixmap = pixmap.scaled(self.image_preview_label.width(), self.image_preview_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_preview_label.setPixmap(scaled_pixmap)
        super().resizeEvent(event)

def run_app():
    app = QApplication(sys.argv)
    
    ex = MultiZooApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
   
    print(f"Uygulama başlatılıyor...")
    print(f"Varsayılan model yolu: {DEFAULT_MODEL_PATH}")
    print(f"Sınıf eşleme dosyası yolu: {os.path.join(config.MODEL_SAVE_DIR, config.CLASS_MAPPING_FILE)}")
    
   
    if not os.path.exists(DEFAULT_MODEL_PATH) or not os.path.exists(os.path.join(config.MODEL_SAVE_DIR, config.CLASS_MAPPING_FILE)):
        print("UYARI: Model dosyası veya sınıf eşleme dosyası varsayılan yolda bulunamadı.")
        print("Lütfen önce modeli eğitin (`python main.py --mode train`) veya dosyaların doğru yerde olduğundan emin olun.")
        
    run_app() 