use std::collections::HashMap;
use std::io::{self, Write};

/// Environment: a map of variable/function names to runtime values
type Env = HashMap<String, Value>;

/// Tokens produced by the tokenizer from input text. Represents the smallest meaningful
/// element of the source code after lexing.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    /// An identifier, used for variable names, function names, operators (+, -), and built-ins.
    Ident(String),
    /// An integer literal, stored as a 32-bit signed integer.
    Number(i32),
    /// A floating-point literal, stored as a 64-bit float.
    Float(f64),
    /// A string literal, supporting escape sequences (e.g., \n, \t).
    Str(String),
    /// The opening parenthesis token: `(`.
    LParen,
    /// The closing parenthesis token: `)`.
    RParen,
    /// The comma separator token: `,`.
    Comma,
}

/// AST node types for parsed expressions. Represents the structure of the program
/// after parsing, forming the Abstract Syntax Tree.
#[derive(Debug, Clone)]
enum Expr {
    /// A literal integer value.
    Number(i32),
    /// A literal floating-point value.
    Float(f64),
    /// An identifier used for variable lookups, function names, or operators.
    Ident(String),
    /// A literal string value.
    Str(String),
    /// A function call or special form (like `let`, `if`, or an operator like `+`).
    Call {
        /// The name of the function or operator being called (e.g., "let", "+").
        name: String,
        /// The vector of arguments, which are themselves expressions to be evaluated.
        args: Vec<Expr>,
    },
}

/// Runtime values the interpreter can hold and operate on. All evaluation results are one of these variants.
#[derive(Debug, Clone)]
enum Value {
    /// A runtime floating-point number.
    Float(f64),
    /// A runtime integer.
    Int(i32),
    /// A runtime string.
    String(String),
    /// A runtime boolean value.
    Boolean(bool),

    /// A function/closure value, capturing its environment for lexical scoping.
    Closure {
        /// The names of the parameters accepted by this function.
        param_names: Vec<String>,
        /// The declared types of the parameters.
        param_types: Vec<Type>,
        /// The declared return type of the function.
        ret_type: Type,
        /// The body of the function, stored as an unevaluated AST expression.
        body: Box<Expr>,
        /// The environment captured at the time the function was defined.
        env: Env, 
    },
}

impl Value {
    /// Checks if two `Value` instances share the exact same enum variant (i.e., the same type),
    /// ignoring the data they hold.
    ///
    /// # Note
    ///
    /// This implementation currently only returns `true` for matching numeric types (`Int` and `Float`)
    /// and `false` for all other type comparisons, including non-numeric types and comparison
    /// between two different numeric types (e.g., Int vs. Float).
    ///
    /// # Arguments
    ///
    /// * `self` - The first `Value` instance.
    /// * `other` - A reference to the second `Value` instance for comparison.
    ///
    /// # Returns
    ///
    /// `true` if both `self` and `other` are the same numeric variant, `false` otherwise.
    pub fn is_same_variant(&self, other: &Value) -> bool {
        // TODO: add capability to make it work with strings and other types like booleans.
        match (self, other) {
            // Works only with specified Value shapes
            (Value::Int(_), Value::Int(_)) => true,
            (Value::Float(_), Value::Float(_)) => true,
            // Fails on everything else
            _ => false,
        }
    }
}

/// Type information (declared/expected) — used for type checking and coercion.
#[derive(Debug, Clone, PartialEq)]
enum Type {
    /// Represents the integer type (`i32`).
    Int,
    /// Represents the floating-point type (`f64`).
    Float,
    /// Represents the boolean type.
    Bool,
    /// Represents the string type.
    String,
    /// A fallback used for unrecognized or generic types.
    Unknown,
}

/// Helper: Converts a string representation of a type (e.g., "int") into the corresponding
/// `Type` enum variant.
///
/// # Arguments
///
/// * `name` - The string slice representing the type name.
///
/// # Returns
///
/// The matching `Type` variant, or `Type::Unknown` if no match is found.
fn parse_type(name: &str) -> Type {
    match name {
        "int" => Type::Int,
        "float" => Type::Float,
        "bool" => Type::Bool,
        "string" => Type::String,
        _ => Type::Unknown,
    }
}

/// How runtime values should be printed in REPL output
impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Int(i) => write!(f, "{}", i),
            Value::Float(fl) => write!(f, "{}", fl),
            Value::String(s) => write!(f, "{}", s),
            Value::Boolean(b) => write!(f, "{}", b),
            Value::Closure {
                param_types,
                ret_type,
                ..
            } => {
                let params_str: Vec<String> =
                    param_types.iter().map(|t| format!("{:?}", t)).collect();
                write!(
                    f,
                    "<closure ({}) -> {:?}>",
                    params_str.join(", "),
                    ret_type
                )
            }
        }
    }
}

/// Binary operator kinds. Used within the `binop` function to dispatch the correct arithmetic operation.
#[derive(Clone, Copy)]
enum BinOpKind {
    /// The addition operation (`+`).
    Add,
    /// The subtraction operation (`-`).
    Sub,
    /// The multiplication operation (`*`).
    Mul,
    /// The division operation (`/`).
    Div,
    /// The modulo operation (`%`).
    Mod,
}

/// The main entry point of the program. Initializes and runs the Read-Eval-Print Loop (REPL).
///
/// # Returns
///
/// An `io::Result<()>` indicating success or any I/O error encountered during the REPL session.
fn main() -> io::Result<()> {
    run_repl()
}

/// Tokenizes the raw input string into a sequence of `Token`s.
/// Handles literals (numbers, floats, strings with escapes), identifiers, and delimiters.
///
/// # Arguments
///
/// * `input` - The raw string input from the user/file.
///
/// # Returns
///
/// A `Vec<Token>` representing the stream of tokens. Prints error messages for unknown characters.
fn tokenize(input: &str) -> Vec<Token> {
    let mut tokens = Vec::new(); // This is the final token variable we'll return
    let mut chars = input.chars().peekable(); // Converts to the Peekable Iterator type (.iter and .peek)

    while let Some(&ch) = chars.peek() {
        match ch {
            // Left parenthesis handling
            '(' => {
                tokens.push(Token::LParen);
                chars.next();
            }
            // Right parenthesis handling
            ')' => {
                tokens.push(Token::RParen);
                chars.next();
            }
            // Comma handling
            ',' => {
                tokens.push(Token::Comma);
                chars.next();
            }

            // Quote handling
            '"' | '\'' => {
                let quote = ch;
                chars.next(); // consume opening quote
                let mut s = String::new();
                while let Some(&c) = chars.peek() {
                    chars.next();
                    // when encountering the other quote, break
                    if c == quote {
                        break;
                    }
                    if c == '\\' {
                        // handle escapes: \" \\ \n \t \r
                        if let Some(&esc) = chars.peek() {
                            chars.next();
                            match esc {
                                'n' => s.push('\n'),
                                't' => s.push('\t'),
                                'r' => s.push('\r'),
                                '\\' => s.push('\\'),
                                '"' => s.push('"'),
                                '\'' => s.push('\''),
                                _ => {
                                    // unknown escape -> keep literally
                                    s.push('\\');
                                    s.push(esc);
                                }
                            }
                        } else {
                            s.push('\\');
                        }
                    } else {
                        s.push(c);
                    }
                }
                // Push the string to the tokens vector
                tokens.push(Token::Str(s));
            }

            // Number or float literal
            '0'..='9' => {
                let mut number = String::new(); // store the number as a string literal during tokenization
                let mut has_dot = false; // flag for when the number contains a decimal component

                while let Some(&c) = chars.peek() { // decimal logic and error checking
                    if c.is_digit(10) {
                        number.push(c);
                        chars.next();
                    } else if c == '.' && !has_dot {
                        has_dot = true;
                        number.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }

                if has_dot {
                    match number.parse::<f64>() {
                        Ok(f) => tokens.push(Token::Float(f)),
                        Err(_) => {
                            eprintln!("Invalid float literal: {}", number);
                        }
                    }
                } else {
                    match number.parse::<i32>() {
                        Ok(i) => tokens.push(Token::Number(i)),
                        Err(_) => {
                            eprintln!("Invalid integer literal: {}", number);
                        }
                    }
                }
            }

            // Identifier (variable or function name)
            c if is_ident_start(c) => {
                let mut ident = String::new();
                while chars.peek().map_or(false, |c| is_ident_char(*c)) {
                    ident.push(chars.next().unwrap());
                }
                tokens.push(Token::Ident(ident));
            }

            // Skip whitespace
            c if c.is_whitespace() => {
                chars.next(); // skip
            }

            // Unknown characters
            _ => {
                println!("Unknown character: {}", ch);
                chars.next(); // skip
            }
        }
    }

    tokens
}

/// Determines if a character is valid as the starting character of an identifier or operator.
///
/// # Arguments
///
/// * `c` - The character to check.
///
/// # Returns
///
/// `true` if the character can start an identifier/operator, `false` otherwise.
fn is_ident_start(c: char) -> bool {
    c.is_alphabetic() || "+-*/%_<>!=.".contains(c)
}

/// Determines if a character is valid for any position (after the start) of an identifier or operator.
///
/// # Arguments
///
/// * `c` - The character to check.
///
/// # Returns
///
/// `true` if the character can be part of an identifier/operator, `false` otherwise.
fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() || "+-*/%_<>!=.".contains(c)
}

/// Parser struct for consuming tokens into an AST
struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    /// Creates a new `Parser` instance, initializing its token stream and position.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The vector of tokens generated by the tokenizer.
    fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
    }

    /// Parses a single top-level expression from the token stream.
    /// Handles literals (`Number`, `Float`, `Str`), identifiers (`Ident`), and function calls (`Call`).
    /// Implements lookahead to distinguish identifiers from function calls.
    ///
    /// # Returns
    ///
    /// An `Option<Expr>` containing the parsed AST node on success, or `None` on a parsing error.
    fn parse_expr(&mut self) -> Option<Expr> {
        match self.peek().cloned()? {
            Token::Number(n) => {
                self.pos += 1;
                Some(Expr::Number(n))
            }
            Token::Float(f) => {
                self.pos += 1;
                Some(Expr::Float(f))
            }
            Token::Str(s) => {
                self.pos += 1;
                Some(Expr::Str(s))
            }
            Token::Ident(name) => {
                let name = name.clone();
                self.pos += 1;

                // Function call: ident '(' args ')'
                if self.peek() == Some(&Token::LParen) {
                    self.pos += 1; // skip '('
                    let mut args = Vec::new();

                    // Parse comma-separated arguments
                    if self.peek() != Some(&Token::RParen) {
                        loop {
                            args.push(self.parse_expr()?);
                            if self.peek() == Some(&Token::Comma) {
                                self.pos += 1;
                            } else {
                                break;
                            }
                        }
                    }

                    // Expect closing ')'
                    if self.peek() != Some(&Token::RParen) {
                        println!("Expected ')'");
                        return None;
                    }

                    self.pos += 1; // skip ')'
                    Some(Expr::Call { name, args })
                } else {
                    Some(Expr::Ident(name))
                }
            }
            _ => {
                println!("Unexpected token: {:?}", self.peek());
                None
            }
        }
    }

    /// Returns a reference to the token at the current position without consuming it.
    ///
    /// # Returns
    ///
    /// An `Option<&Token>` which is `Some` if there are more tokens, or `None` if at the end of the stream.
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }
}

/// Parses a parameter specification from an expression, e.g., `int(x)`.
///
/// # Arguments
///
/// * `expr` - The expression to parse.
///
/// # Returns
///
/// An `Option` containing a tuple of the parameter name (`String`) and its `Type`,
/// or `None` if the expression is not a valid parameter specification.
fn parse_param_spec(expr: &Expr) -> Option<(String, Type)> {
    if let Expr::Call { name, args } = expr {
        if args.len() == 1 {
            if let Expr::Ident(param_name) = &args[0] {
                let param_type = parse_type(name);
                if param_type != Type::Unknown {
                    return Some((param_name.clone(), param_type));
                }
            }
        }
    }
    println!("Invalid parameter specification. Expected `type(name)`, e.g., `int(x)`.");
    None
}

/// Extracts a `Type` from an expression that should be a simple identifier, e.g., `int`.
///
/// # Arguments
///
/// * `expr` - The expression to parse.
///
/// # Returns
///
/// An `Option` containing the `Type`, or `None` if the expression is not a valid type identifier.
fn parse_type_from_expr(expr: &Expr) -> Option<Type> {
    if let Expr::Ident(name) = expr {
        let ty = parse_type(name);
        if ty != Type::Unknown {
            return Some(ty);
        }
    }
    println!("Invalid type expression. Expected a type name like `int`, `float`, etc.");
    None
}

/// Evaluates an AST expression into a runtime `Value`.
/// This is the core interpreter loop, handling special forms (`let`, `func`, `if`),
/// built-in operations, and user-defined function calls (closures).
///
/// # Arguments
///
/// * `expr` - The AST node to evaluate.
/// * `env` - The mutable current environment for variable lookups and assignments.
///
/// # Returns
///
/// An `Option<Value>` containing the result of the evaluation, or `None` if a runtime error occurs.
fn eval(expr: &Expr, env: &mut Env) -> Option<Value> {
    match expr {
        Expr::Number(n) => Some(Value::Int(*n)),
        Expr::Float(f) => Some(Value::Float(*f)),
        Expr::Str(s) => Some(Value::String(s.clone())),

        // Function or special form call
        Expr::Call { name, args } => match name.as_str() {
            // Variable definition: let(name, value)
            "let" => {
                if args.len() != 2 {
                    println!("let() expects 2 arguments");
                    return None;
                }

                // The variable name
                let name_expr = &args[0];
                let value_expr = &args[1];

                // First arg: variable name
                let var_name = match name_expr {
                    Expr::Ident(name) => name.clone(),
                    _ => {
                        println!("First argument to let() must be an identifier");
                        return None;
                    }
                };

                // If second arg is a func(...) definition, store as closure directly
                if let Expr::Call {
                    name: fname,
                    args: _fargs,
                } = value_expr
                {
                    if fname == "func" {
                        let val = eval(value_expr, env)?;
                        env.insert(var_name, val);
                        return None;
                    }
                }

                // Otherwise: expect a typed constructor like int(...)
                let typed_value = match value_expr {
                    Expr::Call {
                        name: type_name,
                        args: inner_args,
                    } => {
                        if inner_args.len() != 1 {
                            println!("Type constructor {}() expects 1 argument", type_name);
                            return None;
                        }

                        let val = eval(&inner_args[0], env)?;
                        match type_name.as_str() {
                            "int" => match val {
                                Value::Float(f) => Some(Value::Int(f as i32)), // temporary cast
                                Value::Int(i) => Some(Value::Int(i)),
                                _ => {
                                    println!("Cannot convert {:?} to int", val);
                                    None
                                }
                            },
                            "float" => match val {
                                Value::Int(i) => Some(Value::Float(i as f64)),
                                Value::Float(f) => Some(Value::Float(f)),
                                _ => {
                                    println!("Cannot convert {:?} to float", val);
                                    None
                                }
                            },
                            "string" => Some(Value::String(format!("{}", val))),
                            "bool" => match val {
                                Value::Boolean(b) => Some(Value::Boolean(b)),
                                _ => {
                                    println!("Cannot convert {:?} to bool", val);
                                    None
                                }
                            },
                            _ => {
                                println!("Unknown type constructor: {}", type_name);
                                None
                            }
                        }
                    }
                    _ => {
                        println!(
                            "Second argument to let() must be a typed constructor call like int(...), float(...), etc."
                        );
                        None
                    }
                }?;

                env.insert(var_name, typed_value);
                None
            }
            // Function definition: func(param1, param2, ..., return_type, body)
            "func" => {
                if args.len() < 2 {
                    println!("func() expects at least 2 arguments: return_type, body");
                    return None;
                }

                let body_expr = args.last().unwrap().clone();
                let return_type_expr = &args[args.len() - 2];
                let param_specs = &args[0..args.len() - 2];

                let mut param_names = Vec::new();
                let mut param_types = Vec::new();

                for spec in param_specs {
                    match parse_param_spec(spec) {
                        Some((name, ty)) => {
                            param_names.push(name);
                            param_types.push(ty);
                        }
                        None => return None,
                    }
                }

                let ret_type = match parse_type_from_expr(return_type_expr) {
                    Some(ty) => ty,
                    None => {
                        println!("Invalid return type specified for func");
                        return None;
                    }
                };

                let captured = env.clone();

                Some(Value::Closure {
                    param_names,
                    param_types,
                    ret_type,
                    body: Box::new(body_expr),
                    env: captured,
                })
            }

            "if" => {
                if args.len() != 3 {
                    println!("if() expects 3 arguments: condition, thenExpr, elseExpr");
                    return None;
                }
                let cond_val = eval(&args[0], env)?;
                match cond_val {
                    Value::Boolean(b) => {
                        if b {
                            eval(&args[1], env)
                        } else {
                            eval(&args[2], env)
                        }
                    }
                    _ => {
                        println!("Type error: if() condition must be a boolean");
                        None
                    }
                }
            }

            "pass" => {
                if !args.is_empty() {
                    println!("pass() takes no arguments");
                }
                None
            }

            // Builtin math functions
            "+" => binop(&args, BinOpKind::Add, env),
            "-" => binop(&args, BinOpKind::Sub, env),
            "*" => binop(&args, BinOpKind::Mul, env),
            "/" => binop(&args, BinOpKind::Div, env),
            "%" => binop(&args, BinOpKind::Mod, env),

            // Equality operators
            "<" => cmp_numeric(&args, |a, b| a < b, env),
            "<=" => cmp_numeric(&args, |a, b| a <= b, env),
            ">" => cmp_numeric(&args, |a, b| a > b, env),
            ">=" => cmp_numeric(&args, |a, b| a >= b, env),
            "==" => eq_op(&args, true, env),
            "!=" => eq_op(&args, false, env),

            // Logical operators
            "and" => logical_and(&args, env),
            "or" => logical_or(&args, env),
            "not" => logical_not(&args, env),
            "xor" => logical_xor(&args, env),

            // Other builtins
            "echo" => expr_echo(&args, env),
            "float" => cast_float(&args, env),
            "len" => len_builtin(&args, env),
            "reverse" => reverse_builtin(&args, env),
            "__dbg_info" => {
                println!("rshell v0.1 by Simon Harms");
                None
            }

            // User defined function calls
            _ => {
                // Try user-defined function from env first
                if let Some(val) = env.get(name) {
                    if let Value::Closure {
                        param_names,
                        param_types,
                        ret_type: _rt,
                        body,
                        env: captured,
                    } = val.clone()
                    {
                        if args.len() != param_names.len() {
                            println!(
                                "function {} expects {} arguments, but got {}",
                                name,
                                param_names.len(),
                                args.len()
                            );
                            return None;
                        }

                        let mut call_env = captured.clone();
                        for (i, arg_expr) in args.iter().enumerate() {
                            let mut arg_val = eval(arg_expr, env)?;
                            let param_type = &param_types[i];

                            arg_val = match (param_type, arg_val) {
                                (Type::Int, v @ Value::Int(_)) => v,
                                (Type::Int, Value::Float(f)) => Value::Int(f as i32),
                                (Type::Float, v @ Value::Float(_)) => v,
                                (Type::Float, Value::Int(i)) => Value::Float(i as f64),
                                (Type::Bool, v @ Value::Boolean(_)) => v,
                                (Type::String, v @ Value::String(_)) => v,
                                (Type::Unknown, v) => v,
                                (pt, v) => {
                                    println!(
                                        "Type error: param {:?} does not accept value {:?}",
                                        pt, v
                                    );
                                    return None;
                                }
                            };
                            call_env.insert(param_names[i].clone(), arg_val);
                        }

                        return eval(&body, &mut call_env);
                    }
                }

                println!("Unknown function: {}", name);
                None
            }
        },

        // Variable lookup
        Expr::Ident(name) => {
            if name == "true" {
                return Some(Value::Boolean(true));
            }
            if name == "false" {
                return Some(Value::Boolean(false));
            }

            env.get(name).cloned().or_else(|| {
                println!("Unknown identifier: {}", name);
                None
            })
        }
    }
}

/// Implements the `echo(...)` built-in function.
/// Evaluates and prints all arguments separated by a space.
///
/// # Returns
///
/// Always returns `None`, as `echo` is side-effectual and does not return a value.
fn expr_echo(args: &[Expr], env: &mut Env) -> Option<Value> {
    for arg in args {
        if let Some(v) = eval(arg, env) {
            print!("{} ", v);
        } else {
            print!("(null) ");
        }
    }
    println!();
    None
}

/// Implements the `float(x)` built-in function.
/// Evaluates `x` and casts the result to a `Value::Float` if it is an Int or Float.
///
/// # Returns
///
/// An `Option<Value::Float>` on successful casting, or `None` on a type error.
fn cast_float(args: &[Expr], env: &mut Env) -> Option<Value> {
    if args.len() != 1 {
        println!("float() takes exactly one argument");
        return None;
    }

    let value = eval(&args[0], env)?;

    match value {
        Value::Int(i) => Some(Value::Float(i as f64)),
        Value::Float(f) => Some(Value::Float(f)), // already float? no-op
        _ => {
            println!("Type error: float() expects an int or float");
            None
        }
    }
}

/// Helper: Checks that all `Value`s in the vector are of the exact same variant (type)
/// as the `expected_type`.
///
/// # Arguments
///
/// * `values` - A slice of evaluated runtime values.
/// * `expected_type` - A sample `Value` instance whose variant is used for comparison.
///
/// # Returns
///
/// `true` if all values share the same `Value` variant, `false` otherwise.
fn all_values_same(values: &Vec<Value>, expected_type: &Value) -> bool {
    values.iter().all(|x| {
        return x.is_same_variant(&expected_type)
    } )
}

/// Handles N-ary binary operations (+, -, *, /, %).
///
/// Evaluates all arguments, performs strict type checking (all must be the same Int or Float),
/// and uses `try_fold` to reduce the list of arguments cumulatively.
/// Includes logic for division-by-zero errors.
///
/// # Arguments
///
/// * `args` - The slice of AST expressions (operands).
/// * `kind` - The type of arithmetic operation to perform.
/// * `env` - The mutable current environment.
///
/// # Returns
///
/// The calculated `Value::Int` or `Value::Float`, or `None` on type/math errors.
fn binop(args: &[Expr], kind: BinOpKind, env: &mut Env) -> Option<Value> {
    // Corrected argument evaluation and collection
    let values: Vec<Value> = args.iter().filter_map(|e| eval(e, env)).collect();

    // --- INITIAL CHECKS ---
    if values.len() < 2 {
        println!("Error: Expected 2 or more arguments.");
        return None;
    }

    let first_value = &values[0];
    
    // Check if the expected type is numeric
    if !(first_value.is_same_variant(&Value::Int(0)) || first_value.is_same_variant(&Value::Float(0.0))) {
        println!("Type error: Operators expect numeric operands.");
        return None;
    }

    // Check uniformity (assuming is_same_variant is correctly implemented)
    if !all_values_same(&values, first_value) {
        println!("Type error: All operands should be the same type.");
        return None;
    }
    // --- END CHECKS ---

    // --- CORE LOGIC: UNPACK, FOLD, AND WRAP ---
    match first_value {
        Value::Int(_) => {
            let mut iter = values.into_iter();
            
            // Unpack initial value (safe due to checks)
            let initial: i32 = if let Value::Int(i) = iter.next().unwrap() { i } else { return None };

            // Fold (reduce) the rest using i32 arithmetic
            let final_result = iter.try_fold(initial, |acc, next_val| {
                let next_num: i32 = if let Value::Int(i) = next_val { i } else { return None }; // Should be safe
                
                match kind {
                    BinOpKind::Add => Some(acc + next_num),
                    BinOpKind::Sub => Some(acc - next_num),
                    BinOpKind::Mul => Some(acc * next_num),
                    BinOpKind::Div => {
                        if next_num == 0 { println!("Math error: Division by zero"); None } 
                        else { Some(acc / next_num) }
                    }
                    BinOpKind::Mod => {
                        if next_num == 0 { println!("Math error: Modulo by zero"); None } 
                        else { Some(acc % next_num) }
                    }
                }
            })?;
            Some(Value::Int(final_result))
        }

        Value::Float(_) => {
            let mut iter = values.into_iter();
            
            // Unpack initial value (safe due to checks)
            let initial: f64 = if let Value::Float(f) = iter.next().unwrap() { f } else { return None };

            // Fold (reduce) the rest using f64 arithmetic
            let final_result = iter.try_fold(initial, |acc, next_val| {
                let next_num: f64 = if let Value::Float(f) = next_val { f } else { return None }; // Should be safe
                
                match kind {
                    BinOpKind::Add => Some(acc + next_num),
                    BinOpKind::Sub => Some(acc - next_num),
                    BinOpKind::Mul => Some(acc * next_num),
                    BinOpKind::Div => {
                        if next_num == 0.0 { println!("Math error: Division by zero"); None } 
                        else { Some(acc / next_num) }
                    }
                    BinOpKind::Mod => {
                        if next_num == 0.0 { println!("Math error: Modulo by zero"); None } 
                        else { Some(acc % next_num) }
                    }
                }
            })?;
            Some(Value::Float(final_result))
        }

        // All other non-numeric cases are caught by the initial checks
        _ => None,
    }
}

/// Helper: Performs numeric comparisons (<, <=, >, >=) on exactly two arguments.
/// Handles mixed Int and Float types by promoting Ints to Floats for comparison.
///
/// # Arguments
///
/// * `args` - The slice of AST expressions (operands). Must contain exactly two elements.
/// * `cmp` - A generic closure function (`Fn(f64, f64) -> bool`) defining the comparison logic.
/// * `env` - The mutable current environment for evaluating arguments.
///
/// # Returns
///
/// An `Option<Value::Boolean>` containing the result of the comparison, or `None` on a type or length error.
fn cmp_numeric<F>(args: &[Expr], cmp: F, env: &mut Env) -> Option<Value>
where
    F: Fn(f64, f64) -> bool,
{
    if args.len() != 2 {
        println!("Comparison expects 2 arguments");
        return None;
    }
    let a = eval(&args[0], env)?;
    let b = eval(&args[1], env)?;
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Some(Value::Boolean(cmp(x as f64, y as f64))),
        (Value::Float(x), Value::Float(y)) => Some(Value::Boolean(cmp(x, y))),
        (Value::Int(x), Value::Float(y)) => Some(Value::Boolean(cmp(x as f64, y))),
        (Value::Float(x), Value::Int(y)) => Some(Value::Boolean(cmp(x, y as f64))),
        _ => {
            println!("Type error: numeric comparison requires numbers");
            None
        }
    }
}

/// Helper: Performs equality comparisons (==, !=) on exactly two arguments.
/// Supports comparison between numeric, string, and boolean types, including mixed Int/Float comparisons.
///
/// # Arguments
///
/// * `args` - The slice of AST expressions (operands). Must contain exactly two elements.
/// * `expect_equal` - A boolean flag: `true` for '==', `false` for '!='.
/// * `env` - The mutable current environment for evaluating arguments.
///
/// # Returns
///
/// An `Option<Value::Boolean>` containing the result of the equality check, or `None` on a type or length error.
fn eq_op(args: &[Expr], expect_equal: bool, env: &mut Env) -> Option<Value> {
    if args.len() != 2 {
        println!("Equality expects 2 arguments");
        return None;
    }
    let a = eval(&args[0], env)?;
    let b = eval(&args[1], env)?;

    let eq = match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => x == y,
        (Value::Int(x), Value::Float(y)) => (x as f64) == y,
        (Value::Float(x), Value::Int(y)) => x == (y as f64),
        (Value::String(s1), Value::String(s2)) => s1 == s2,
        (Value::Boolean(b1), Value::Boolean(b2)) => b1 == b2,
        _ => {
            println!("Type error: cannot compare these types for equality");
            return None;
        }
    };

    Some(Value::Boolean(if expect_equal { eq } else { !eq }))
}

/// Helper: Evaluates an expression and asserts that the result is a `Value::Boolean`, returning its inner `bool`.
///
/// # Arguments
///
/// * `expr` - The AST expression to evaluate.
/// * `env` - The mutable current environment.
///
/// # Returns
///
/// An `Option<bool>` containing the result on success, or `None` if the evaluation fails or the result is not a Boolean.
fn eval_bool(expr: &Expr, env: &mut Env) -> Option<bool> {
    match eval(expr, env)? {
        Value::Boolean(b) => Some(b),
        other => {
            println!("Type error: expected bool, got {:?}", other);
            None
        }
    }
}

/// Implements the logical AND operator, evaluating exactly two boolean expressions.
/// Performs short-circuit evaluation (returns false immediately if the first argument is false).
///
/// # Arguments
///
/// * `args` - The slice of AST expressions. Must contain exactly two elements.
/// * `env` - The mutable current environment.
///
/// # Returns
///
/// An `Option<Value::Boolean>` containing the result, or `None` on a type or length error.
fn logical_and(args: &[Expr], env: &mut Env) -> Option<Value> {
    if args.len() != 2 {
        println!("and() expects 2 arguments");
        return None;
    }
    let a = eval_bool(&args[0], env)?;
    if !a {
        return Some(Value::Boolean(false)); // short-circuit
    }
    let b = eval_bool(&args[1], env)?;
    Some(Value::Boolean(b))
}

/// Implements the logical OR operator, evaluating exactly two boolean expressions.
/// Performs short-circuit evaluation (returns true immediately if the first argument is true).
///
/// # Arguments
///
/// * `args` - The slice of AST expressions. Must contain exactly two elements.
/// * `env` - The mutable current environment.
///
/// # Returns
///
/// An `Option<Value::Boolean>` containing the result, or `None` on a type or length error.
fn logical_or(args: &[Expr], env: &mut Env) -> Option<Value> {
    if args.len() != 2 {
        println!("or() expects 2 arguments");
        return None;
    }
    let a = eval_bool(&args[0], env)?;
    if a {
        return Some(Value::Boolean(true)); // short-circuit
    }
    let b = eval_bool(&args[1], env)?;
    Some(Value::Boolean(b))
}

/// Implements the logical NOT operator, evaluating exactly one boolean expression.
///
/// # Arguments
///
/// * `args` - The slice of AST expressions. Must contain exactly one element.
/// * `env` - The mutable current environment.
///
/// # Returns
///
/// An `Option<Value::Boolean>` containing the inverted result, or `None` on a type or length error.
fn logical_not(args: &[Expr], env: &mut Env) -> Option<Value> {
    if args.len() != 1 {
        println!("not() expects 1 argument");
        return None;
    }
    let a = eval_bool(&args[0], env)?;
    Some(Value::Boolean(!a))
}

/// Implements the logical XOR operator, evaluating exactly two boolean expressions.
///
/// # Arguments
///
/// * `args` - The slice of AST expressions. Must contain exactly two elements.
/// * `env` - The mutable current environment.
///
/// # Returns
///
/// An `Option<Value::Boolean>` containing the result, or `None` on a type or length error.
fn logical_xor(args: &[Expr], env: &mut Env) -> Option<Value> {
    if args.len() != 2 {
        println!("xor() expects 2 arguments");
        return None;
    }
    let a = eval_bool(&args[0], env)?;
    let b = eval_bool(&args[1], env)?;
    Some(Value::Boolean(a ^ b))
}

/// Implements the `len` built-in, returning the character length of a string.
///
/// # Arguments
///
/// * `args` - The slice of AST expressions. Must contain exactly one element that evaluates to a `Value::String`.
/// * `env` - The mutable current environment.
///
/// # Returns
///
/// An `Option<Value::Int>` containing the length, or `None` on a type or length error.
fn len_builtin(args: &[Expr], env: &mut Env) -> Option<Value> {
    if args.len() != 1 {
        eprintln!("len() expects 1 argument");
        return None;
    }
    match eval(&args[0], env)? {
        Value::String(s) => Some(Value::Int(s.chars().count() as i32)),
        other => {
            eprintln!("TypeError: len() expects string, got {:?}", other);
            None
        }
    }
}

/// Implements the `reverse(s)` built-in, returning the reversed string.
///
/// # Arguments
///
/// * `args` - The slice of AST expressions. Must contain exactly one element that evaluates to a `Value::String`.
/// * `env` - The mutable current environment.
///
/// # Returns
///
/// An `Option<Value::String>` containing the reversed string, or `None` on a type or length error.
fn reverse_builtin(args: &[Expr], env: &mut Env) -> Option<Value> {
    if args.len() != 1 {
        eprintln!("reverse() expects 1 argument");
        return None;
    }
    match eval(&args[0], env)? {
        Value::String(s) => Some(Value::String(s.chars().rev().collect())),
        other => {
            eprintln!("TypeError: reverse() expects string, got {:?}", other);
            None
        }
    }
}

/// Runs the interactive Read-Eval-Print Loop (REPL).
/// Handles printing the prompt, reading input, checking for 'exit', and passing input to `evaluate_line`.
///
/// # Returns
///
/// An `io::Result<()>` indicating success or any I/O error (reading or writing) during the session.
fn run_repl() -> io::Result<()> {
    let mut stdout = io::stdout(); // stdout variable to write to
    let mut input = String::new(); // input to evaluate
    let mut env = Env::new(); // holds the environment of the sessions like variables and function definitions

    loop {
        stdout.write_all(b"rsh$ ")?;
        stdout.flush()?;

        input.clear();
        io::stdin().read_line(&mut input)?;

        let trimmed = input.trim();

        if trimmed == "exit" {
            break;
        }

        evaluate_line(trimmed.to_string(), &mut env)?;

        stdout.write_all(trimmed.as_bytes())?;
        stdout.write_all(b"\n")?;
    }

    Ok(())
}

/// Parses and evaluates a single line of REPL input.
/// Coordinates the pipeline: Tokenize -> Parse -> Eval. Prints the final result or an error message.
///
/// # Arguments
///
/// * `input` - The trimmed string input from the REPL line.
/// * `env` - The mutable, persistent global environment.
///
/// # Returns
///
/// An `io::Result<()>` for I/O operations (though it primarily handles logic).
fn evaluate_line(input: String, env: &mut Env) -> io::Result<()> {
    let tokens = tokenize(&input);
    let mut parser = Parser::new(tokens);

    if let Some(expr) = parser.parse_expr() {
        if let Some(result) = eval(&expr, env) {
            println!("{}", result);
        }
    } else {
        println!("Could not parse input.");
    }

    Ok(())
}
