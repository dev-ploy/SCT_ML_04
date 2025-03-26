from model import create_model
from data_loader import load_data
from tensorflow.keras.optimizers import Adam

def train_model(data_dir, epochs):
    images, labels = load_data(data_dir)
    model = create_model()
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(images, labels, epochs=epochs)

if __name__ == "__main__":
    train_model('data/processed/', epochs=10)