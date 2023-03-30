use std::{
    collections::HashMap,
    fmt::{Display, Formatter},
};

// -------------------------------------------------------------------------------------------------

fn main() {
    let expr = gen_rand_expr();
    println!("{expr}\n");

    let ssa = Compiler::compile(&expr);
    println!("{ssa}\n");

    let expr_result = expr.eval();
    println!("EXPR RESULT: {expr_result}");

    let ssa_result = ssa.eval();
    println!("EXPR RESULT: {expr_result}");

    println!(
        "\n{}",
        if expr_result == ssa_result {
            "SUCCESS"
        } else {
            "FAILURE"
        }
    );
}

// -------------------------------------------------------------------------------------------------
// Simple expression.

#[derive(Debug)]
enum Expr {
    Number(i64),
    Add(Box<Expr>, Box<Expr>),
    Cond(Box<Expr>, Box<Expr>, Box<Expr>),
}

impl Display for Expr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Number(n) => write!(f, "{n}"),
            Expr::Add(l, r) => write!(f, "(+ {l} {r})"),
            Expr::Cond(c, i, e) => write!(f, "(if {c} {i} {e})"),
        }
    }
}

impl Expr {
    fn eval(&self) -> i64 {
        match self {
            Expr::Number(n) => *n,
            Expr::Add(l, r) => l.eval() + r.eval(),
            Expr::Cond(c, i, e) => {
                if c.eval() != 0 {
                    i.eval()
                } else {
                    e.eval()
                }
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------

fn gen_rand_expr() -> Expr {
    fn helper(depth: u64) -> Box<Expr> {
        let new_number = || Expr::Number(rand::random::<i64>() % 10);
        let new_arith_op = || Expr::Add(helper(depth + 1), helper(depth + 1));
        let new_cond = || Expr::Cond(helper(depth + 1), helper(depth + 1), helper(depth + 1));

        let r = rand::random::<i64>() % 100;
        Box::new(if depth < 2 {
            if r < 5 {
                new_cond()
            } else {
                new_arith_op()
            }
        } else if depth < 4 {
            new_arith_op()
        } else {
            new_number()
        })
    }

    *helper(0)
}

// -------------------------------------------------------------------------------------------------
// SSA representation.

#[derive(Debug)]
struct Ssa {
    blocks: Vec<Block>,
}

impl Display for Ssa {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for (i, b) in self.blocks.iter().enumerate() {
            write!(f, "\nblock{i}:\n{b}")?;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct Block {
    instrs: Vec<Instruction>,
}

type BlockIdx = usize;

impl Display for Block {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for i in &self.instrs {
            write!(f, "    {i}\n")?;
        }
        Ok(())
    }
}

#[derive(Debug)]
enum Instruction {
    Add(Register, Register, Register),
    Br(BlockIdx),
    Cbr(Register, BlockIdx, BlockIdx),
    Move(Register, Register),
    Movi(Register, Const),
    Ret(Register),
}

impl Display for Instruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Add(d, l, r) => write!(f, "add {d} {l} {r}"),
            Instruction::Br(i) => write!(f, "br block{i}"),
            Instruction::Cbr(c, tr, fa) => write!(f, "cbr {c} block{tr} block{fa}"),
            Instruction::Move(d, s) => write!(f, "mov {d} {s}"),
            Instruction::Movi(d, c) => write!(f, "mov {d} {}", c.0),
            Instruction::Ret(r) => write!(f, "ret {r}"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Const(i64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Register(u64);

impl Display for Register {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.0)
    }
}

impl Ssa {
    fn eval(&self) -> i64 {
        let mut regs = HashMap::new();
        self.eval_block(&mut regs, 0)
    }

    fn eval_block(&self, regs: &mut HashMap<Register, i64>, block_idx: BlockIdx) -> i64 {
        self.blocks[block_idx]
            .instrs
            .iter()
            .map(|instr| match instr {
                Instruction::Add(d, l, r) => {
                    let result = regs.get(&l).unwrap() + regs.get(&r).unwrap();
                    regs.insert(*d, result);
                    result
                }
                Instruction::Br(i) => self.eval_block(regs, *i),
                Instruction::Cbr(c, tr, fa) => {
                    if *regs.get(&c).unwrap() != 0 {
                        self.eval_block(regs, *tr)
                    } else {
                        self.eval_block(regs, *fa)
                    }
                }
                Instruction::Move(d, s) => {
                    let result = *regs.get(&s).unwrap();
                    regs.insert(*d, result);
                    result
                }
                Instruction::Movi(d, c) => {
                    regs.insert(*d, c.0);
                    c.0
                }
                Instruction::Ret(r) => *regs.get(r).unwrap(),
            })
            .last()
            .unwrap()
    }
}

// -------------------------------------------------------------------------------------------------

struct Compiler {
    blocks: Vec<Block>,
    cur_block_idx: usize,

    next_reg_idx: u64,
}

// -------------------------------------------------------------------------------------------------

impl Compiler {
    fn compile(expr: &Expr) -> Ssa {
        let mut compiler = Compiler::new();

        let result_reg = compiler.compile_expr(expr);
        compiler.append_instr(Instruction::Ret(result_reg));

        Ssa {
            blocks: compiler.blocks,
        }
    }

    fn new() -> Self {
        let blocks = vec![Block { instrs: Vec::new() }];
        Compiler {
            blocks,
            cur_block_idx: 0,
            next_reg_idx: 0,
        }
    }

    fn compile_expr(&mut self, expr: &Expr) -> Register {
        match expr {
            Expr::Number(n) => {
                let instr_reg = self.next_reg();
                self.append_instr(Instruction::Movi(instr_reg, Const(*n)));
                instr_reg
            }
            Expr::Add(l, r) => {
                let l_reg = self.compile_expr(l);
                let r_reg = self.compile_expr(r);
                let instr_reg = self.next_reg();
                self.append_instr(Instruction::Add(instr_reg, l_reg, r_reg));
                instr_reg
            }
            Expr::Cond(c, t, f) => {
                let c_reg = self.compile_expr(c);
                let c_end_block_idx = self.cur_block_idx;

                let t_begin_block_idx = self.new_block();
                let t_reg = self.compile_expr(t);
                let t_end_block_idx = self.cur_block_idx;

                let f_begin_block_idx = self.new_block();
                let f_reg = self.compile_expr(f);
                let f_end_block_idx = self.cur_block_idx;

                self.append_instr_to(
                    c_end_block_idx,
                    Instruction::Cbr(c_reg, t_begin_block_idx, f_begin_block_idx),
                );

                let j_block_idx = self.new_block();
                let instr_reg = self.next_reg();
                self.append_instr_to(t_end_block_idx, Instruction::Move(instr_reg, t_reg));
                self.append_instr_to(t_end_block_idx, Instruction::Br(j_block_idx));
                self.append_instr_to(f_end_block_idx, Instruction::Move(instr_reg, f_reg));
                self.append_instr_to(f_end_block_idx, Instruction::Br(j_block_idx));

                instr_reg
            }
        }
    }

    fn new_block(&mut self) -> BlockIdx {
        let idx = self.blocks.len();
        self.blocks.push(Block { instrs: Vec::new() });
        self.cur_block_idx = idx;
        idx
    }

    fn append_instr(&mut self, instr: Instruction) {
        self.append_instr_to(self.cur_block_idx, instr)
    }

    fn append_instr_to(&mut self, block_idx: usize, instr: Instruction) {
        self.blocks[block_idx].instrs.push(instr);
    }

    fn next_reg(&mut self) -> Register {
        let reg_idx = self.next_reg_idx;
        self.next_reg_idx += 1;
        Register(reg_idx)
    }
}

// -------------------------------------------------------------------------------------------------
