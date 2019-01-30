from __future__ import print_function
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd
from genetic_selection import GeneticSelectionCV


def main():


    df = pd.read_csv('kddcup.data_10_percent_corrected')

#print(df)


    df.flag = pd.Categorical(df.flag)
    df['flag'] = df.flag.cat.codes

    df.protocol_type = pd.Categorical(df.protocol_type)
    df['protocol_type'] = df.protocol_type.cat.codes

    df.service = pd.Categorical(df.service)
    df['service'] = df.service.cat.codes

    df.attack = pd.Categorical(df.attack)
    df['attack'] = df.attack.cat.codes

    target = df['attack']
    df = df.drop(['attack'],axis=1)
    
    
    # Some noisy data not correlated
    E = np.random.uniform(0, 0.1, size=(len(df), 20))

    X = np.hstack((df, E))
    y = target

    estimator = linear_model.LogisticRegression()

    selector = GeneticSelectionCV(estimator,
                                  cv=5,
                                  verbose=1,
                                  scoring="accuracy",
                                  n_population=50,
                                  crossover_proba=0.5,
                                  mutation_proba=0.2,
                                  n_generations=40,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  caching=True,
                                  n_jobs=-1)
    selector = selector.fit(X, y)

    print(selector.support_)


if __name__ == "__main__":
    main()
