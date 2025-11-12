# rshell

An interpreted programming language featuring static types and a unique functional syntax.

## Building

To build the project, navigate to the root directory of the project and run:

```bash
cargo build
```

## Running

To run the project, use the following command:

```bash
cargo run
```

That will bring you into a REPL. Everything in rshell functions as a function and there is no syntax other than functions and data (no keywords) and that is done by design to make it simple.

## Commands

### Built-in functions and keywords

 * `let(name, type(value))` - Variable and function declaration
 * `func(param1, param2, ..., return_type, body)` - Defines a new function (technically an anonymous function)
 * `if(condition), then_expr, else_expr)` - Conditional expression
 * `pass()` - A no-op that does nothing

### Arithmetic
 * Note that the order of operations is most nested to least nested
 * `+(a, b, ...)` - Add two or more numbers
 * `-(a, b, ...)` - Subtract two or more numbers
 * `*(a, b, ...)` - Multiply two or more numbers
 * `/(a, b, ...)` - Divide two or more numbers
 * `%(a, b, ...)` - Add two or more numbers

### Comparison
 * `<(a, b)` - Less than
 * `<=(a, b)` - Less than or equal to
 * `>(a, b)` - More than
 * `>=(a, b)` - More than or equal to
 * `==(a, b)` - Equal to
 * `!=(a, b)` - Not equal to

### Logical
 * `and(a, b)` - Logical AND
 * `or(a, b)` - Logical OR
 * `not(a)` - Logical NOT
 * `xor(a, b)` - Logical XOR

### Utility functions
 * `echo(...)` - prints the argument to the console
 * `float(x)` - casts an integer to a float
 * `len(string)` - Returns the length of a string
 * `reverse(string)` - Reverses a string

## Examples

```
let(square, func(int(x), int, *(x, x)))
let(cube, func(int(x), int, *(x, x, x)))
```
You can type `exit` to leave the REPL.