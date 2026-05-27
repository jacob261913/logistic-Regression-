import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('machine_failure_data.csv')
# Preprocess the data
label_encoder = LabelEncoder()

data["Type_encoded"] = label_encoder.fit_transform(data["Type"])

X = data[["Type_encoded","Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "TWF", "HDF", "PWF", "OSF", "RNF"]]
Y = data["Machine failure"]

model = LogisticRegression(max_iter=1000)
model.fit(X, Y)
print("Coeff=", model.coef_)
print("Intercept=", model.intercept_)

GIVEN_INPUT =[[1,301,311,1450,52,28,1,0,0,0,0]]

prediction = model.predict(GIVEN_INPUT)

print("Prediction for the given input is:", prediction)

if prediction[0] == 1:
    print("The machine will fail.")
else:    print("No Failure, The machine is not likely to fail.")








