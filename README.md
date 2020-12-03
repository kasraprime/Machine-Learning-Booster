# Machine-Learning-Booster

A template to implement Machine Learning projects faster.

You need to define the following based on your project:

1. Your model in a file named "models.py", and the class name which can be anything. I set it to TheModel.
2. Your dataset loader in a file named "datasets.py" and a function called data_loader
3. analyze function in which you can save model's ouput and make some plots and csv files.
4. compute_metrics function in order to compute metrics you want including f1 score and accuracy

Run the following command:

```bash
bash run_ML.sh develop 200 100 attention train-test random 42 tanh
```
