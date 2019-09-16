from sklearn.linear_model import LogisticRegression
from data_handler import load_data, prepare_data

input_dir = '../input'

X, y, labels = prepare_data(load_data(input_dir))

model = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000)
model.fit(X, y)
prediction = model.predict_proba(X)[:, 1]

print(prediction.shape)
print(prediction[240:250])
