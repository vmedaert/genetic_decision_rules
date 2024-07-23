from sklearn.model_selection import train_test_split
'''arff: https://pythonhosted.org/liac-arff/'''
import arff
from pickle import dump
import os
from sklearn.metrics import recall_score, precision_score, f1_score
import argparse
from joblib import Parallel, delayed
import time
from genetic_decision_rules import process_arff, create_initial_population, train_interpretability, train_accuracy_interpretability, accuracy

@delayed
def main(
    in_path,
    out_dir,
    file_type,
    pop_size,
    pop_inc,
    max_rule_size,
    max_rule_count,
    mutation_prob,
    max_epochs,
    timeout,
    output_name,
    output_space,
    test_size,
    accuracy_score="accuracy",
    interpretability_score="size",
    crowding_distance=False,
    initialisation="random_forest",
    preprocess=True
):
    # load the data (only arff files are supported atm)
    arff_data = arff.load(open(in_path))
    if output_space is not None:
        arff_data["attributes"][-1] = (output_name, output_space)
    data, spaces, encoding = process_arff(arff_data, output_name=output_name)
    encoding.attribute_names = [name for name, _ in spaces[:-1]]
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    start_time = time.time()
    # generate an initial population (random forest or random + simple genetic algorithm)
    if initialisation == "random_forest":
        initial_population = create_initial_population(
            X_train,
            y_train,
            spaces,
            pop_size=pop_size,
            max_rule_size=max_rule_size,
            max_rule_count=max_rule_count
        )
    else:
        initial_population = [encoding.decode(encoding.random_chromosome(max_rule_count)) for _ in range(pop_size)]
    if preprocess:
        par, _ = train_interpretability(
            encoding,
            initial_population,
            X_train,
            y_train,
            max_epochs=400
        )
        i = len(initial_population) - 1
        for _, rulelist in par:
            if i < 0:
                break
            initial_population[i] = rulelist
            i -= 1
    end_time = time.time()


    # train a rulelist and output the results
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output = train_accuracy_interpretability(
        encoding,
        initial_population,
        X_train,
        y_train,
        max_epochs=max_epochs,
        timeout=timeout,
        pop_inc=pop_inc,
        mutation_prob=mutation_prob,
        track_history=True,
        accuracy_score=accuracy_score,
        interpretability_score=interpretability_score,
        crowding_distance=crowding_distance
    )
    with open(f"{out_dir}/output.pkl", "wb") as file:
        dump(output["population"], file)
    pareto_set = sorted(output["population"][0], key=lambda x : x[0][0])
    with open(f"{out_dir}/pareto_set.csv", "w") as file:
        for vec, chrom in pareto_set:
            file.write(f"{','.join((str(e) for e in vec))}\n{','.join((str(e) for e in chrom))}\n")
    with open(f"{out_dir}/pareto_set.txt", "w") as file:
        for vec, chrom in pareto_set:
            file.write(f"vector: {vec}\n\n")
            rulelist = encoding.decode(chrom)
            file.write(f"{rulelist}\n\n")
            file.write(f"accuracy on test set is {accuracy(rulelist, X_test, y_test)}\n\n")
            res = [rulelist.apply(row) for row in X_test]
            recall = recall_score(y_test, res, labels=encoding.output_space, average=None, zero_division=0)
            precision = precision_score(y_test, res, labels=encoding.output_space, average=None, zero_division=0)
            f1 = f1_score(y_test, res, labels=encoding.output_space, average=None, zero_division=0)
            for cls, rec, pre, f1s in zip(encoding.output_space, recall, precision, f1):  
                # the positive labels that are correctly classified
                file.write(f"'{cls}' recall on test set is {rec}\n")
                # the positive classified instances that are correct
                file.write(f"'{cls}' precision on test set is {pre}\n")
                # harmonic mean of recall and precision
                file.write(f"'{cls}' f1 score on test set is {f1s}\n\n")
            file.write(f"{'-'*60}\n")
    with open(f"{out_dir}/info.txt", "w") as file:
        file.write("## attributes\n")
        file.write("# maximum amount of epochs\n")
        file.write(f"max_epochs = {max_epochs}\n")
        file.write("# every timeout amount of epochs the population is checked for change, if no change -> stop execution\n")
        file.write(f"timeout = {timeout}\n")
        file.write("# amount of new individuals that is generated every epoch using crossover\n")
        file.write(f"pop_inc = {pop_inc}\n")
        file.write("# probability of substitution\n")
        file.write(f"mutation_prob = {mutation_prob}\n")
        file.write("# population size\n")
        file.write(f"pop_size = {pop_size}\n")
        file.write("# maximum size of rules\n")
        file.write(f"max_rule_size = {max_rule_size}\n")
        file.write("# maximum amount of rules\n")
        file.write(f"max_rule_count = {max_rule_count}\n")
        file.write("## notes\n")
        file.write(f"preprocessing time: {round(end_time - start_time)}s\n")
        file.write(f"ran {output['epochs']} epochs in {output['time']} seconds\n")
        file.write(f"total execution time: {output['time'] + round(end_time - start_time)}s")
    with open(f"{out_dir}/pareto_history.csv", "w") as file:
        file.write(
            "\n".join((f"{epoch}," + 
                ",".join((
                    ",".join([str(e) for e in vec]) for vec in pareto_set
                )) for epoch, pareto_set in output["pareto_history"]
            ))
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train a rulelist"
    )
    parser.add_argument(
        "in_path",
        help="path to the input file"
    )
    parser.add_argument(
        "out_dir",
        help="directory to write output files to"
    )
    parser.add_argument(
        "--file_type",
        help="the type of the input file",
        choices=("arff",),
        default="arff"
    )
    parser.add_argument(
        "--pop_size",
        help="population size",
        type=int,
        action="append",
        default=[]
    )
    parser.add_argument(
        "--pop_inc",
        help="amount of new individuals that is generated every epoch using crossover",
        type=int,
        action="append",
        default=[]
    )
    parser.add_argument(
        "--max_rule_size",
        help="maximum size of rules",
        type=int,
        default=6
    )
    parser.add_argument(
        "--max_rule_count",
        help="maximum amount of rules",
        type=int,
        default=15
    )
    parser.add_argument(
        "--mutation_prob",
        help="probability of substitution",
        type=float,
        action="append",
        default=[]
    )
    parser.add_argument(
        "--max_epochs",
        help="maximum amount of epochs",
        type=int,
        default=10000
    )
    parser.add_argument(
        "--timeout",
        help="every timeout amount of epochs the population is checked for change, if no change -> stop execution",
        type=int,
        default=100
    )
    parser.add_argument(
        "--output_name",
        help="name of the output field in the data",
        default="class"
    )
    parser.add_argument(
        "--output_space",
        help="the output space can be set here if not correctly set in the input file",
        action="append"
    )
    parser.add_argument(
        "--n_runs",
        help="the amount of times to train the model with the same parameters",
        type=int,
        default=1
    )
    parser.add_argument(
        "--test_size",
        help="relative size of the test set",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--accuracy_score",
        help="'accuracy', 'balanced_accuracy' or 'mean_f1'",
        choices=("accuracy", "balanced_accuracy", "mean_f1"),
        default="accuracy"
    )
    parser.add_argument(
        "--interpretability_score",
        help="'size' or 'size_overlap'",
        choices=("size", "size_overlap"),
        default="size"
    )
    parser.add_argument(
        "--start_id",
        help="id of the first run",
        type=int,
        default=0
    )
    parser.add_argument(
        "--crowding_distance",
        help="additionally use crowding distance when sorting the population",
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--initialisation",
        help="type of initialisation of the population",
        choices=("random", "random_forest"),
        default="random_forest"
    )
    parser.add_argument(
        "--preprocess",
        help="if true, use a single-objective genetic algorithm for a few epochs",
        action=argparse.BooleanOptionalAction
    )
    args = parser.parse_args()
    if len(args.pop_size) == 0:
        args.pop_size.append(100)
    if len(args.pop_inc) == 0:
        args.pop_inc.append(50)
    if len(args.mutation_prob) == 0:
        args.mutation_prob.append(0.01)
    parallel = Parallel(n_jobs=-1)
    parallel(
        main(
            args.in_path,
            f"{args.out_dir}/ps{pop_size}pi{pop_inc}mp{int(mutation_prob*1000)}/run_{i}",
            args.file_type,
            pop_size,
            pop_inc,
            args.max_rule_size,
            args.max_rule_count,
            mutation_prob,
            args.max_epochs,
            args.timeout,
            args.output_name,
            args.output_space,
            test_size=args.test_size,
            accuracy_score=args.accuracy_score,
            interpretability_score=args.interpretability_score,
            crowding_distance=args.crowding_distance is not None,
            initialisation=args.initialisation,
            preprocess=args.preprocess
        )
        for mutation_prob in args.mutation_prob
        for pop_inc in args.pop_inc
        for pop_size in args.pop_size
        for i in range(args.start_id, args.start_id + args.n_runs)
    )