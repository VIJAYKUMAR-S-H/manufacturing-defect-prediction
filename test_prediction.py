from src.predict import PredictionPipeline

sample = {
    "Injection_Temperature": 220,
    "Injection_Pressure": 55,
    "Cycle_Time": 40,
    "Cooling_Time": 12,
    "Material_Viscosity": 180,
    "Ambient_Temperature": 28,
    "Machine_Age": 4,
    "Operator_Experience": 5,
    "Maintenance_Hours": 120,
    "Shift": "Morning",
    "Machine_Type": "Hydraulic",
    "Material_Grade": "A",
    "Day_of_Week": "Monday",
    "Temperature_Pressure_Ratio": 4.0,
    "Total_Cycle_Time": 52,
    "Efficiency_Score": 90,
    "Machine_Utilization": 82
}

pipeline = PredictionPipeline()

prediction = pipeline.predict(sample)

print(f"\nPredicted Parts Per Hour: {prediction:.2f}")