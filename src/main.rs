use std::collections::HashMap;
use std::io::{self, Write};

/// Environment: a map of variable/function names to runtime values
type Env = HashMap<String, Value>;

/// Tokens produced by the tokenizer from input text
#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String), // e.g. "+", "__dbg_info"
    Number(i32),   // e.g. 42
    Float(f64),    // e.g. 42.42
    Str(String),
    LParen, // (
    RParen, // )
    Comma,  // ,
}

/// AST node types for parsed expressions
#[derive(Debug, Clone)]
enum Expr {
    Number(i32),
    Float(f64),
    Ident(String),
    Str(String),
    Call { name: String, args: Vec<Expr> },
}

/// Runtime values the interpreter can hold and operate on
#[derive(Debug, Clone)]
enum Value {
    Float(f64),
    Int(i32),
    String(String),
    Boolean(bool),

    /// A function/closure value
    /// - Stores the parameter name, its type, the return type,
    ///   the function body, and the captured environment
    Closure {
        param_name: String,
        param_type: Type,
        ret_type: Type,
        body: Box<Expr>,
        env: Env, // captured env at definition time
    },
}

/// Type information (declared/expected) — used for type checking/coercion
#[derive(Debug, Clone, PartialEq)]
enum Type {
    Int,
    Float,
    Bool,
    String,
    Unknown, // fallback for unrecognized types
}

/// Helper: convert a string like "int" into a Type enum
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
                param_name,
                param_type,
                ret_type,
                ..
            } => {
                write!(
                    f,
                    "<closure {}: {:?} -> {:?}>",
                    param_name, param_type, ret_type
                )
            }
        }
    }
}

/// Binary operator kinds
#[derive(Clone, Copy)]
enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

fn main() -> io::Result<()> {
    run_repl()
}

/// Tokenize raw input string into a sequence of Tokens
fn tokenize(input: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            '(' => {
                tokens.push(Token::LParen);
                chars.next();
            }
            ')' => {
                tokens.push(Token::RParen);
                chars.next();
            }
            ',' => {
                tokens.push(Token::Comma);
                chars.next();
            }

            '"' | '\'' => {
                let quote = ch;
                chars.next(); // consume opening quote
                let mut s = String::new();
                while let Some(&c) = chars.peek() {
                    chars.next();
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
                tokens.push(Token::Str(s));
            }

            // Number or float literal
            '0'..='9' => {
                let mut number = String::new();
                let mut has_dot = false;

                while let Some(&c) = chars.peek() {
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
                            println!("Invalid float literal: {}", number);
                        }
                    }
                } else {
                    match number.parse::<i32>() {
                        Ok(i) => tokens.push(Token::Number(i)),
                        Err(_) => {
                            println!("Invalid integer literal: {}", number);
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

/// First char of an identifier must be letter or symbol like + - * / % _
fn is_ident_start(c: char) -> bool {
    c.is_alphabetic() || "+-*/%_<>!=.".contains(c)
}

/// Rest of identifier may also contain digits
fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() || "+-*/%_<>!=.".contains(c)
}

/// Parser struct for consuming tokens into an AST
struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
    }

    /// Parse an expression: number, float, identifier, or function call
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

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }
}

/// Evaluate an AST expression into a runtime Value
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
            // Function definition: func(int(x), int(<body>))
            "func" => {
                // func(...(x), ...(body)) where ... is a type
                if args.len() != 2 {
                    println!("func() expects at least 2 arguments: param_spec, return_typed(body)");
                    return None;
                }

                // Param spec: e.g., int(x)
                let (param_name, param_ty) = match &args[0] {
                    Expr::Call {
                        name: ty_name,
                        args: inner,
                    } => {
                        if inner.len() != 1 {
                            println!(
                                "paramater spec {}() expectes exactly one identifier",
                                ty_name
                            );
                            return None;
                        }
                        let pname = match &inner[0] {
                            Expr::Ident(s) => s.clone(),
                            _ => {
                                println!("paramater spec must be like int(x) with identifier");
                                return None;
                            }
                        };
                        (pname, parse_type(ty_name))
                    }
                    _ => {
                        println!("First argument to func must be a param spec like int(x)");
                        return None;
                    }
                };

                // Return type wrapper: e.g., int(*(x, x))
                let (ret_ty, body_expr) = match &args[1] {
                    Expr::Call {
                        name: ty_name,
                        args: inner,
                    } => {
                        if inner.len() != 1 {
                            println!(
                                "return type wrapper {}() expects one body expression",
                                ty_name
                            );
                            return None;
                        }
                        (parse_type(ty_name), Box::new(inner[0].clone()))
                    }
                    _ => {
                        println!(
                            "Second argument to func() must be return_typed(body), e.g., int(<expr)"
                        );
                        return None;
                    }
                };

                // Capture current env (closure)
                let captured = env.clone();

                Some(Value::Closure {
                    param_name,
                    param_type: param_ty,
                    ret_type: ret_ty,
                    body: body_expr,
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
                        param_name,
                        param_type,
                        ret_type: _rt,
                        body,
                        env: captured,
                    } = val.clone()
                    {
                        if args.len() != 1 {
                            println!("function {} expects 1 argument", name);
                            return None;
                        }

                        // Evaluate arg
                        let mut arg_val = eval(&args[0], env)?;

                        // Coerce or reject based on param type
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

                        // New env for call: captured + arg binding
                        let mut call_env = captured.clone();
                        call_env.insert(param_name, arg_val);

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

/// echo(...) built-in — prints values separated by spaces
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

/// float(x) built-in — casts int to float
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

/// Binary operation helper for +, -, *, /, %
fn binop(args: &[Expr], kind: BinOpKind, env: &mut Env) -> Option<Value> {
    if args.len() != 2 {
        println!("Expected 2 or more arguments");
        return None;
    } // Originally this only worked with 2 arguments, that's why it is named binop

    //println!("{:?}", args);
    let a = eval(&args[0], env)?;
    let b = eval(&args[1], env)?;

    match (a, b) {
        (Value::Int(x), Value::Int(y)) => {
            let result = match kind {
                BinOpKind::Add => x + y,
                BinOpKind::Sub => x - y,
                BinOpKind::Mul => x * y,
                BinOpKind::Div => {
                    if y == 0 {
                        println!("Math error: division by zero");
                        return None;
                    }
                    x / y
                }
                BinOpKind::Mod => {
                    if y == 0 {
                        println!("Math error: modulo by zero");
                        return None;
                    }
                    x % y
                }
            };
            Some(Value::Int(result))
        }

        (Value::Float(x), Value::Float(y)) => {
            let result = match kind {
                BinOpKind::Add => x + y,
                BinOpKind::Sub => x - y,
                BinOpKind::Mul => x * y,
                BinOpKind::Div => {
                    if y == 0.0 {
                        println!("Math error: division by zero");
                        return None;
                    }
                    x / y
                }
                BinOpKind::Mod => {
                    if y == 0.0 {
                        println!("Math error: modulo by zero");
                        return None;
                    }
                    x % y
                }
            };
            Some(Value::Float(result))
        }

        _ => {
            println!("Type error: cannot operate on mixed or non-numeric types");
            None
        }
    }
}

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

fn eval_bool(expr: &Expr, env: &mut Env) -> Option<bool> {
    match eval(expr, env)? {
        Value::Boolean(b) => Some(b),
        other => {
            println!("Type error: expected bool, got {:?}", other);
            None
        }
    }
}

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

fn logical_not(args: &[Expr], env: &mut Env) -> Option<Value> {
    if args.len() != 1 {
        println!("not() expects 1 argument");
        return None;
    }
    let a = eval_bool(&args[0], env)?;
    Some(Value::Boolean(!a))
}

fn logical_xor(args: &[Expr], env: &mut Env) -> Option<Value> {
    if args.len() != 2 {
        println!("xor() expects 2 arguments");
        return None;
    }
    let a = eval_bool(&args[0], env)?;
    let b = eval_bool(&args[1], env)?;
    Some(Value::Boolean(a ^ b))
}

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

fn run_repl() -> io::Result<()> {
    let mut stdout = io::stdout();
    let mut input = String::new();
    let mut env = Env::new();

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

/// Run parser + evaluator for one line of REPL input
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
