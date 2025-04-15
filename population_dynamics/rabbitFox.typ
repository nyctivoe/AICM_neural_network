#set text(
  font: "Libertinus Serif",
  size: 11pt
)

#set page(
  paper: "us-letter",
  margin: (x: 2.54cm, y: 2.54cm),
  numbering: "1",
)

#set par(
  justify: true,
  leading: 0.52em,
)

#align(center, text(17pt)[
    *Population Dynamics*\
    Spencer Wang\
    AICM, Period 7\
    Princeton International School of Mathematics and Science
])

#set heading(numbering: "1.1 -")

= Exercise 1

== Analysis

We can then analyze what happens when all parameters a, b, c, d are the same. Meaning that the predation of fox on rabbit, rabbit's reproduction rate, and the competition between foxes have the same strength (although doesn't make sense irl). 

#line(length: 100%)
```python
prey_initial = 10
predator_initial = 5
a = 1.0
b = 1.0
c = 1.0
d = 1.0

rf = RabbitFoxes(prey_init=prey_initial, predator_init=predator_initial,
                 a=a, b=b, c=c, d=d, seed=42)

rf.run_simulation(
    max_t=15,
    num_points=1000,
    num_initial_conditions=5,
    prey_range=(10, 100),
    predator_range=(2, 20)
)

```
#line(length: 100%)


#figure(image("ex1/1.png"))
#figure(image("ex1/2.png"))

Here if we plot properly, we can see that the system will get absolutely destroyed. The rabbit population will go to 0, and then the fox population will also go to 0. Thus, ggs. Notice that the initial condition of the rabbit-fox system do indeed mater (a bit). Meaning, two different initial values will give us two different cycling patterns.

#line(length: 100%)
```python
prey_initial = 10
predator_initial = 5
a = 5.0
b = 0.5
c = 0.7
d = 0.01

rf = RabbitFoxes(prey_init=prey_initial, predator_init=predator_initial,
                 a=a, b=b, c=c, d=d, seed=42)

rf.run_simulation(
    max_t=15,
    num_points=1000,
    num_initial_conditions=5,
    prey_range=(10, 100),
    predator_range=(2, 20)
)

```
#line(length: 100%)

#figure(image("ex1/3.png"))
#figure(image("ex1/4.png"))

Here parameters are what they are. The meaning is rather intuitive so yeah. Here, this is a nice example of a cycling system. There is no point of "equilibrium" in this system. The rabbit and fox populations will just keep cycling. 

However, the initial conditions do matter. If we change the initial conditions, the cycling pattern will change. And actually, will decide wether the system will go to extinction or not (not mathematically, but to our human eyes). 

Overall, the parameters are really sensitive to changes. Like some minor tweaks to variable numbers will change the outputs drastically.

== Unrealistic

This system is pretty unrealistic if we think about it. Like how are the rabbit population just unbounded? Also, I don't believe the interaction term can be simply treated as a linear term. But it's a nice example of a simple system that can be solved analytically and demonstrated clearly with a graph.

Like look at this...

#line(length: 100%)
```python
prey_initial = 10
predator_initial = 5
a = 0.2
b = 1.0
c = 1.0
d = 1.0

rf = RabbitFoxes(prey_init=prey_initial, predator_init=predator_initial,
                 a=a, b=b, c=c, d=d, seed=42)

rf.run_simulation(
    max_t=15,
    num_points=1000,
    num_initial_conditions=1,
    prey_range=(100, 100),
    predator_range=(0, 0)
)
```
#line(length: 100%)

#figure(image("ex1/5.png"))
#figure(image("ex1/6.png"))

== Fix

So we can just make the first term of the rabbit bounded by a logistic growth term. This is a simple fix to the problem.

#line(length: 100%)
```py
def lotka_volterra(self, state, t):
    """
    Defines the Lotka-Volterra equations with logistic growth for prey.

    Parameters:
    - state (list): Current state [prey, predator].
    - t (float): Current time (unused, but required by odeint).

    Returns:
    - list: Derivatives [dprey_dt, dpredator_dt].
    """
    prey, predator = state
    dprey_dt = self.a * prey * (1 - prey / self.K) - self.b * prey * predator
    dpredator_dt = self.d * prey * predator - self.c * predator
    return [dprey_dt, dpredator_dt]
```
#line(length: 100%)

We can then try the model with the following parameters:

#line(length: 100%)
```python
a = 5.0
b = 0.5
c = 0.7
d = 0.01
K = 200

rf = RabbitFoxes(a=a, b=b, c=c, d=d, K=K, seed=42)

rf.run_simulation(
    max_t=50,
    num_points=1000,
    num_initial_conditions=5,
    prey_range=(15, 50),
    predator_range=(3, 10)
)
```
#line(length: 100%)

#figure(image("ex1/7.png"))
#figure(image("ex1/8.png"))

Now we can observe that they all enter a fixed point, which is good (now the whole thing is stable). The rabbit population is bounded by the carrying capacity, and the fox population is bounded by the rabbit population. This is a more realistic model of the system. This basically stops the previously observed trend where the rabbit will increase without bounds and then foxes just start to massacre them. Then the number of foxes got way too out of hand and then started to kill off too many rabbits, leading to an unstable system.s

== Non Dimensionalization

We can follow the following steps to non-dimensionalize the system:

#figure(image("ex5/solution1.jpg"))

Ofcourse, there are more than one way to non-dimensionalize this system. With my initial setup and choice of reduction towards the end, I ended up with these. 