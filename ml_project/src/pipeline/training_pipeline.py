from src.components.data_loader import load_data
from src.components.trainer import train

def main():
    X_train, X_test, y_train, y_test = load_data("data/predictive_maintenance.csv")
    train(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
