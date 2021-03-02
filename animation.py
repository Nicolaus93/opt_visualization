import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import streamlit as st
import streamlit.components.v1 as components
import torch
from coin_betting import Cocob
from magdir import Magdir
from mirror_descent import MirrorDescent
from regralizer import Regralizer
from utils import *

plt.style.use('seaborn-white')


def draw(i, X, Y, pathline, point):
    x = X[i]
    y = Y[i]
    pathline[0].set_data(X[:i + 1], Y[:i + 1])
    point[0].set_data(x, y)
    return pathline[0], point[0]


def random_trajectory(n):
    # Fixing random state for reproducibility
    np.random.seed(24)
    trajectory = np.zeros((n, 2))
    trajectory[0] = [-1, 1]
    for i in range(1, n):
        trajectory[i][0] = trajectory[i - 1][0] + np.random.rand() * 0.1
        trajectory[i][1] = trajectory[i - 1][1] + np.random.rand() * 0.1
    return trajectory[:, 0], trajectory[:, 1]


def execute_steps(func, initial_state, optimizer_class, optimizer_config, num_iter=500):
    """
    Run the optimizer.
    """
    x = torch.Tensor(initial_state).requires_grad_(True)
    if optimizer_class == Cocob:
        # x = torch.Tensor([0, 0]).requires_grad_(True)
        # optimizer = optimizer_class([x], eps=optimizer_config['lr'])
        optimizer = optimizer_class([x], eps=1.6)
    elif optimizer_class == Magdir:
        optimizer = optimizer_class([x], eps=1)
    elif optimizer_class == MirrorDescent:
        optimizer = optimizer_class([x], diam=optimizer_config['lr'])
    elif optimizer_class == Regralizer:
        optimizer = optimizer_class([x])
    else:
        optimizer = optimizer_class([x], **optimizer_config)
    steps = np.zeros((2, num_iter + 1))
    steps[:, 0] = np.array(initial_state)
    for i in range(1, num_iter + 1):
        optimizer.zero_grad()
        f = func(x)
        f.backward(create_graph=True, retain_graph=True)
        torch.nn.utils.clip_grad_norm_(x, 1.0)
        optimizer.step()
        steps[:, i] = x.detach().numpy()
    return steps


def frame_selector_ui():
    st.sidebar.markdown("# Parameters")

    fun = st.sidebar.radio(
        "Function to optimize:",
        ("Rosenbrock", "Rastrigin"),
    )

    iterations = st.sidebar.slider(
        "iterations:", 500, 1000, step=100)

    # The user can pick which type of object to search for.
    algo = st.sidebar.selectbox(
        "Which algo?", ["SGD", "Adam", "Cocob", "Adagrad", "Magdir", "Mirror Descent", "Regralizer"], 2)

    if algo in ['Cocob', 'Magdir']:
        # Choose initial wealth
        param = st.sidebar.slider(
            "Choose initial wealth:", 0.1, 1.5, value=0.1, step=0.1)
    elif algo is 'Mirror Descent':
        param = st.sidebar.slider(
            "Choose diameter:", 0.1, 10.0, step=0.1)
    else:
        param = st.sidebar.slider(
            "Learning rate:", 0.1, 1.5, value=0.1, step=0.1)

    return fun, algo, param, iterations


def main():
    st.title("Visualize Optimizers in 1d")
    description = st.markdown(
        "In this app we run some optimizers on nasty functions like these ones:")

    # load image of rosenbrock function
    rosenbrock_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Rosenbrock_function.svg/1200px-Rosenbrock_function.svg.png'
    rastrigin_url = 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Rastrigin_function.png'
    im1 = st.image(rosenbrock_url, caption='Rosenbrock function', use_column_width=True)
    im2 = st.image(rastrigin_url, caption='Rastrigin function', use_column_width=True)

    # col1, col2 = st.beta_columns(2)
    # with col1:
    #     im1 = st.image(rosenbrock_url, caption='Rosenbrock function', use_column_width=True)
    # with col2:
    #     im2 = st.image(rastrigin_url, caption='Rastrigin function', use_column_width=True)

    # Add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do:")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode", ["Show instructions", "Run optimizers", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run optimizers".')
    elif app_mode == "Show the source code":
        description.empty()
        # image.empty()
        im1.empty()
        im2.empty()
        with open("animation.py") as f:
            content = f.readlines()
            st.code(''.join(content))
    elif app_mode == "Run optimizers":
        function, selected_algo, selected_learning_rate, iterations = frame_selector_ui()
        if st.sidebar.button('Run!'):
            description.empty()
            # image.empty()
            im1.empty()
            im2.empty()
            run_the_app(function, selected_algo, selected_learning_rate, iterations)


def run_the_app(function, selected_algo, selected_learning_rate, iterations):

    n = 100

    #  create figure
    if function == 'Rosenbrock':
        fig, ax = plot_rosenbrock()
        initial_state = (-2.0, 2.0)
    else:
        fig, ax = plot_rastrigin()
        initial_state = (-2.0, 3.5)

    if selected_algo is None:
        st.error("Please select a different algorithm.")
        return

    optimizers = {'Cocob': Cocob, 'SGD': torch.optim.SGD, 'Adam': torch.optim.Adam,
                  'Adagrad': torch.optim.Adagrad, 'Magdir': Magdir,
                  'Mirror Descent': MirrorDescent, 'Regralizer': Regralizer}
    functions = {'Rosenbrock': rosenbrock, 'Rastrigin': rastrigin}

    # run algo
    algorithm = optimizers[selected_algo]
    function = functions[function]
    lr = selected_learning_rate
    tot_iter = iterations
    steps = execute_steps(function, initial_state, algorithm, {'lr': lr}, num_iter=tot_iter)

    # function animation
    selected_points = np.linspace(0, tot_iter, n).astype(int)
    X = steps[0, :]
    Y = steps[1, :]
    pathline = ax.plot(X[0], Y[0], color='r', lw=1)
    point = ax.plot(X[0], Y[0], "ro")
    point_ani = animation.FuncAnimation(fig, draw, frames=n, fargs=(X, Y, pathline, point),
                                        interval=100, blit=True, repeat=False)

    # video rendering
    with st.spinner('Wait for it...'):
        with open("myvideo.html", "w") as f:
            print(point_ani.to_html5_video(), file=f)
    st.success('Done!')
    st.markdown("Green point is the minimum:")
    HtmlFile = open("myvideo.html", "r")
    source_code = HtmlFile.read()
    components.html(source_code, height=900, width=900)


if __name__ == "__main__":
    main()
