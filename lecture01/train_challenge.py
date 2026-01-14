"""
Lecture 1: Hi, automatic differentiation!

Demonstration: Train a teacher--student linear regression model with gradient
descent.

Learning objectives:

* more jax.numpy
* introducing functional model API
* introducing jax.grad
"""

import time
import tyro
import matthewplotlib as mp
from jaxtyping import Float, Array

import jax
import jax.numpy as jnp


def main(
    num_steps: int = 400,
    learning_rate: float = 0.01,
):
    
    # define the teacher
    w_teacher = jnp.array([0.5, -1.0]) # slope, intercept

    # define the student
    w_student = jnp.array([-1.0, 3.0])

    print(vis(w_student, w_teacher, step=0, loss=jnp.inf))
    
    for step in range(1, num_steps+1):
        grad_fnc = jax.value_and_grad(loss, argnums=[0,1])
        l, (g_student, g_teacher) = grad_fnc(w_student, w_teacher)
        w_student = w_student - learning_rate * g_student
        w_teacher = w_teacher - learning_rate * g_teacher
        plot = vis(w_student, w_teacher, step=step, loss=l)
        print(f"\x1b[{plot.height}A{plot}")

def loss(
        w_student: Float[Array, "2"],
        w_teacher: Float[Array, "2"]
    ) -> float:
    x = jnp.linspace(-4, 4, 80)
    y_student = forward(w_student, x)
    y_teacher = forward(w_teacher, x)
    squared_errors = (y_teacher - y_student) ** 2
    return jnp.mean(squared_errors)
    
def forward(
        w: Float[Array, "2"],
        x: Float[Array, "batch_size"]
    ) -> Float[Array, "batch_size"]:

    a, b = w
    y = a * x + b
    return y


def vis(
    w_student: Float[Array, "2"],
    w_teacher: Float[Array, "2"],
    step: int,
    loss: float,
) -> mp.plot:
    x = jnp.linspace(-4, 4, 80)
    return mp.axes(
        mp.scatter(
            mp.xaxis(-4, 4, 80),
            mp.yaxis(-4, 4, 80),
            (x, forward(w_teacher, x), "cyan"),
            (x, forward(w_student, x), "magenta"),
            height=20,
            width=40,
            xrange=(-4,4),
            yrange=(-4,4),
        ),
        title=f"step {step:03d} | loss {loss:6.3f}",
        ylabel="y",
        xlabel="x",
    )

if __name__ == "__main__":
    tyro.cli(main)
