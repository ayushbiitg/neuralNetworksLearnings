# For Backpropogation_micrograd_from_scratch notebook

Understanding `.grad` in micrograd

---
The one idea everything rests on: the chain rule
Every `Value` object stores two numbers that matter for gradients:
`.data` → the actual value, computed going forward (left → right in the graph).
`.grad` → how much the final output `L` changes if we nudge this node slightly,
computed going backward (right → left).
Formally, for any node `n`:
```
n.grad = ∂L/∂n
```
where `L` is the final output of the whole graph. The gradient of a node is its
influence on the final result — not on the next node.
The chain rule lets us pass that influence backward one hop at a time:
```
∂L/∂n = (∂L/∂out) · (∂out/∂n)
         ^^^^^^^^^   ^^^^^^^^^
         out.grad    local derivative
      (already known)  (each op knows this about itself)
```
In plain English: (how much `out` affects `L`) × (how much `n` affects `out`).
---
How the code encodes this: `_backward`
Each operation defines a `_backward` closure that does exactly one thing:
take `out.grad` and route it to the inputs' `.grad` using the local derivative.
```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0                 # starts at 0 → "no influence known yet"
        self._backward = lambda: None   # default: do nothing (leaf nodes)
        self._prev = set(_children)     # who created me
        self._op = _op
```
---
Example 1 — `__mul__`
```python
def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad  += other.data * out.grad
        other.grad += self.data  * out.grad
    out._backward = _backward

    return out
```
Forward: `out = self * other`.
Local derivatives:
```
∂out/∂self  = other
∂out/∂other = self
```
If `out = a * b`, wiggling `a` changes `out` at a rate of `b`, and wiggling `b`
changes `out` at a rate of `a`.
Chain rule step:
```
self.grad += other.data * out.grad
             ^^^^^^^^^^   ^^^^^^^^
             local deriv  influence flowing in
```
> **Signature of multiplication in backprop:** the two inputs **swap values** and
> each multiplies by the incoming gradient.
---
Example 2 — `exp`
```python
def exp(self):
    x = self.data
    out = Value(math.exp(x), (self,), 'exp')

    def _backward():
        self.grad += out.data * out.grad
    out._backward = _backward

    return out
```
Forward: `out.data = e^(self.data)`.
Local derivative — the beautiful one:
```
d/dx (e^x) = e^x
```
The derivative of `exp` is itself, and we already computed `e^x` — it's sitting in
`out.data`. So the code reuses it instead of recomputing:
```
self.grad += out.data * out.grad
             ^^^^^^^^   ^^^^^^^^
             = e^x      ∂L/∂out
             = ∂out/∂self
```
---
Two subtle rules that trip everyone up
1. Why `+=` and not `=`
A node can feed into multiple places downstream. Each path contributes some
influence on `L`, and the total gradient is the sum of all paths (multivariable
chain rule). Starting `.grad = 0` and accumulating with `+=` handles this
automatically — using `=` would lose contributions from reused variables.
2. The seed: `out.grad = 1.0`
Backprop needs a starting point. The final node is `L`, and:
```
∂L/∂L = 1
```
So before running backprop you set `L.grad = 1.0`. That "1" flows backward, getting
multiplied by local derivatives at each hop.
---
The full backward pass (topological order)
Individual `_backward` calls only work if a node's `out.grad` is finished before we
use it. That requires processing nodes right-to-left in dependency order — a
topological sort:
```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)

    self.grad = 1.0                 # seed ∂L/∂L = 1
    for node in reversed(topo):     # right → left
        node._backward()            # each hop routes gradient to its inputs
```
---
The thinking process — a repeatable recipe
When you meet any new operation and need to write its `_backward`, ask these 4
questions in order:
What's the forward formula? Write `out = f(inputs)`.
What's the local derivative of `out` w.r.t. each input? (Pure calculus,
ignore the rest of the graph.)
Chain it: each input's grad gets `+= (local derivative) * out.grad`.
Accumulate, don't overwrite (`+=`), because inputs may be used elsewhere.
Quick self-test
Operation	Local derivative(s)	`_backward` result
`out = a + b`	`1`, `1`	passes gradient through unchanged (a "router")
`out = a * b`	`b`, `a`	the swap
`out = a ** 2`	`2a`	`a.grad += 2*a.data * out.grad`
`out = exp(a)`	`e^a` (= `out.data`)	`a.grad += out.data * out.grad`
---
TL;DR
> **Forward computes numbers. Backward multiplies local slopes by the incoming
> gradient, and we sum over all paths.**
Once this clicks, the `Value` class stops looking like magic.
