import numpy as np
import sympy as sp
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    dim = "2D"
    mass = "lowmass"

    # Load CSV files
    file1 = pd.read_csv(
        f"/home/anava/projects/MvSR-analysis/toy_results/degeneracy_paths_{dim}_{mass}/perfect/max5/MvSR_results.csv"
    )
    file2 = pd.read_csv(
        f"/home/anava/projects/MvSR-analysis/toy_results/degeneracy_paths_{dim}_{mass}/perfect/max9/MvSR_results.csv"
    )
    file3 = pd.read_csv(
        f"/home/anava/projects/MvSR-analysis/toy_results/degeneracy_paths_{dim}_{mass}/perfect/max15/MvSR_results.csv"
    )
    file4 = pd.read_csv(
        f"/home/anava/projects/MvSR-analysis/toy_results/degeneracy_paths_{dim}_{mass}/perfect/max25/MvSR_results.csv"
    )

    # Combine into a single DataFrame for ease of processing
    df = pd.concat([file1, file2, file3, file4], ignore_index=True)
    df = df.drop_duplicates(subset=["expression"])

    # Function to parse the losses
    def parse_losses(losses):
        return list(map(float, losses.strip("[]").split(",")))

    # Apply parsing to the losses column
    df["losses"] = df["losses"].apply(parse_losses)

    # Calculate summary statistics
    summary_stats = df["losses"].apply(
        lambda x: pd.Series({"mean": np.mean(x), "median": np.median(x), "std_dev": np.std(x)})
    )

    # Add the summary statistics to the original DataFrame
    df = pd.concat([df, summary_stats], axis=1)

    df["DoF"] = df["expression"].str.count(r"[A-Z]")

    # get data list
    data_list = []
    for file_num in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        data_file = f"/home/anava/projects/symbolic_regression_examples/data/paths/{dim}/{mass}/{mass}_{file_num}.txt"
        data_list.append(np.loadtxt(data_file))

    file_nums = [0, 1]
    X1 = []
    X2 = []
    labels = []

    for i in file_nums:
        data = data_list[i]

        X1.append(data[:, 0])
        X2.append(data[:, 1])

        if i == 0:
            labels.append(np.full(data[:, 1].shape, np.e))  # First set of labels are e
        else:
            labels.append(np.full(data[:, 1].shape, np.pi**2))  # Second set of labels are pi^2

    X1 = np.concatenate(X1)
    X2 = np.concatenate(X2)
    labels = np.concatenate(labels)

    variables = sp.symbols("X1 X2")

    function_mapping = {
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "cbrt": lambda x: x ** (sp.S(1) / 3),  # Cube root function
    }

    xlabel = "$\eta$"
    ylabel = "$\chi_{eff}$"

    plot_bounds = [[0.0826, 0.25], [-1, 1]]
    grid_size = 1000
    if plot_bounds == None:
        x_min, x_max = data_list[0][:, 0].min(), data_list[0][:, 0].max()
        y_min, y_max = data_list[0][:, 1].min(), data_list[0][:, 1].max()
    else:
        x_min, x_max = plot_bounds[0]
        y_min, y_max = plot_bounds[1]
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # load expression as function... This means that we need to fit the constant...
    mean_rmse_list = []
    params_list = []
    expressions_list = []  # delete later

    for i, expr_string in enumerate(df["expression"]):  # 114 long here.
        # compute loss then for the expression with the smallest mean loss, we will generate a plot.
        constants = sorted(set(re.findall(r"\b[A-Z]\b", expr_string)))

        constants = sp.symbols(" ".join(constants))

        expr = sp.sympify(expr_string, locals=function_mapping)
        func = sp.lambdify((*variables, *constants), expr, "numpy")

        initial_guess = [0.1] * len(constants)

        def objective(X, *params):
            x1, x2 = X
            return func(x1, x2, *params)

        # params, params_covariance = curve_fit(objective, (X1, X2), labels, p0=initial_guess)
        try:
            # Attempt to fit the curve
            params, params_covariance = curve_fit(objective, (X1, X2), labels, p0=initial_guess)
            # Optionally compute and print/store any statistics, e.g., mean squared error
            print(f"Fit successful for expression {i}: {expr_string}")
        except RuntimeError as e:
            # Handle any errors that occur during fitting
            print(f"Fit failed for expression {i}: {expr_string}")
            continue
        except Exception as e:
            # Handle other exceptions
            print(f"An unexpected error occurred for expression {i}: {expr_string}, error: {str(e)}")
            continue

        # predicted_labels = objective((X1, X2), *params)

        zz = objective((xx.ravel(), yy.ravel()), *params).reshape(xx.shape)

        # zz = model.predict(grid).reshape(xx.shape)

        rmse_list = []

        for i, data in enumerate(data_list):

            # predictions = model.predict((data))
            predictions = objective((data[:, 0], data[:, 1]), *params)
            c_level = np.median(predictions[~np.isnan(predictions)])  # changed from mean for stability.

            # print(c_level, model.predict((data)))
            contours = plt.contour(xx, yy, zz, levels=[c_level], colors="k", linestyles="solid", linewidths=2)

            contour_paths = contours.collections[0].get_paths()
            if len(contour_paths) == 0:
                print("No contours found")
                rmse_list.append(20)
                continue
            contour_points = np.vstack([p.vertices for p in contour_paths])
            # print("contours ",contour_points, data[:, 1], len(contour_points),len(data[:,1]))

            errors = []
            for point in data:
                closest_contour_point = contour_points[np.argmin(np.abs(contour_points[:, 0] - point[0]))]
                y_distance_error = np.abs(closest_contour_point[1] - point[1])
                errors.append(y_distance_error)

            rmse = np.sqrt(mean_squared_error(np.zeros_like(errors), errors))
            rmse_list.append(rmse)

            # print(f"RMSE between points and contour in y-direction: {rmse:.4f}")
            plt.scatter(data[:, 0], data[:, 1], c="blue", label=f"path {i}, RMSE y-dir: {rmse:.4f}")

        print(rmse_list, np.mean(rmse_list))
        mean_rmse_list.append(np.mean(rmse_list))
        params_list.append(params)
        expressions_list.append(expr_string)
        print(f"Mean RMSE between points and contour in y-direction: {np.mean(rmse_list):.4f}")

    # for the best one, we should plot it.
    min_index = mean_rmse_list.index(min(mean_rmse_list))
    print(
        "optimal rmse, index, expression and params: ",
        min(mean_rmse_list),
        min_index,
        expressions_list[min_index],
        params_list[min_index],
    )

    # print(len(df["expression"]), list(df["expression"]))
    expr_string = expressions_list[min_index]  # list(df["expression"])[min_index]
    print(expr_string)
    # expr_string = df.iloc[min_index, "expression"]

    # compute loss then for the expression with the smallest mean loss, we will generate a plot.
    constants = sorted(set(re.findall(r"\b[A-Z]\b", expr_string)))

    constants = sp.symbols(" ".join(constants))

    expr = sp.sympify(expr_string, locals=function_mapping)
    func = sp.lambdify((*variables, *constants), expr, "numpy")

    initial_guess = [0.1] * len(constants)

    def objective(X, *params):
        x1, x2 = X
        return func(x1, x2, *params)

    params = params_list[min_index]
    zz = objective((xx.ravel(), yy.ravel()), *params).reshape(xx.shape)

    # zz = model.predict(grid).reshape(xx.shape)

    rmse_list = []

    plt.clf()
    plt.close("all")
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(xx, yy, zz, shading="auto", cmap="viridis")
    plt.colorbar(label="Function Value")
    for i, data in enumerate(data_list):

        # predictions = model.predict((data))
        predictions = objective((data[:, 0], data[:, 1]), *params)
        c_level = np.median(predictions[~np.isnan(predictions)])  # changed from mean for stability.

        # print(c_level, model.predict((data)))
        contours = plt.contour(xx, yy, zz, levels=[c_level], colors="k", linestyles="solid", linewidths=2)

        # print("CONTOURS!!!!", contours, c_level, model.predict((data)))

        contour_paths = contours.collections[0].get_paths()
        contour_points = np.vstack([p.vertices for p in contour_paths])
        # print("contours ",contour_points, data[:, 1], len(contour_points),len(data[:,1]))

        errors = []
        for point in data:
            closest_contour_point = contour_points[np.argmin(np.abs(contour_points[:, 0] - point[0]))]
            y_distance_error = np.abs(closest_contour_point[1] - point[1])
            errors.append(y_distance_error)
            plt.scatter(closest_contour_point[0], closest_contour_point[1], c="yellow")

        rmse = np.sqrt(mean_squared_error(np.zeros_like(errors), errors))
        rmse_list.append(rmse)
        formatted_errors = map("{:.4f}".format, errors)
        string = ", ".join(formatted_errors)
        print(f"RMSE between points and contour in y-direction: {rmse:.4f} errors: {string}")
        plt.scatter(data[:, 0], data[:, 1], c="blue", label=f"path {i}, RMSE y-dir: {rmse:.4f}")

    plt.title(f"Heatmap of Multi Function Fit MvSR (avg RMSE y-dir: {np.mean(rmse_list):.4f})")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    formatted_params = map("{:.4f}".format, params)
    string = ", ".join(formatted_params)
    plt.legend(title=f"Model: {expr_string} \n {string}", loc="best")

    plt.show()

    import os

    plots_path = "/home/anava/projects/symbolic_regression_examples/data/plots/May28/MultipleFit"
    file_name = f"MvSR_{dim}_{mass}_contour_refs.png"
    os.makedirs(plots_path, exist_ok=True)
    plot_filename = os.path.join(plots_path, file_name)
    plt.savefig(plot_filename)


if __name__ == "__main__":
    main()
