from src.components.evaluator import predict

def main():
    preds = predict("data/predictive_maintenance.csv")
    print("Predictions:", preds)

if __name__ == "__main__":
    main()
