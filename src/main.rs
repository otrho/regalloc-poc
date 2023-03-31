use std::{
    collections::{HashMap, HashSet},
    fmt::{Display, Formatter},
};

// -------------------------------------------------------------------------------------------------

fn main() {
    let expr = gen_rand_expr();
    println!("{expr}\n");

    let mut ssa = Compiler::compile(&expr);
    println!("{ssa}\n");

    let expr_result = expr.eval();
    println!("EXPR RESULT: {expr_result}");

    let ssa_result = ssa.eval();
    println!("SSA RESULT: {expr_result}");

    if let Err(msg) = ssa.reg_alloc() {
        println!("REG ALLOC FAILED: {msg}");
        return;
    }

    let asm_result = ssa.eval();
    println!("ASM RESULT: {asm_result}");

    println!(
        "\n{}",
        if expr_result == ssa_result && ssa_result == asm_result {
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
// SSA representation.  It's not strictly SSA since we don't have PHI nodes nor block params so PHI
// values are def'd from multiple blocks before branching to a common successor.

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

    fn reg_alloc(&mut self) -> Result<(), String> {
        use regalloc2::{PReg, RegClass};

        let env = regalloc2::MachineEnv {
            preferred_regs_by_class: [
                (16..64).map(|i| PReg::new(i, RegClass::Int)).collect(),
                Vec::new(),
            ],
            non_preferred_regs_by_class: [Vec::new(), Vec::new()],
            fixed_stack_slots: Vec::new(),
        };

        let opts = regalloc2::RegallocOptions {
            verbose_log: true,
            validate_ssa: true,
        };

        let ssa_meta = SsaMeta::new(self);
        let _alloc_output = regalloc2::run(&ssa_meta, &env, &opts).map_err(|e| format!("{e}"))?;

        Ok(())
    }
}

struct SsaMeta {
    num_insts: usize,
    num_blocks: usize,
    entry_block: regalloc2::Block,
    block_instr_ranges: HashMap<regalloc2::Block, regalloc2::InstRange>,
    block_succs: HashMap<usize, Vec<regalloc2::Block>>,
    block_preds: HashMap<usize, Vec<regalloc2::Block>>,
    dummy_block_params: Vec<regalloc2::VReg>,
    ret_set: HashSet<usize>,
    branch_set: HashSet<usize>,
    moves: HashMap<usize, (regalloc2::Operand, regalloc2::Operand)>,
    inst_operands: HashMap<usize, Vec<regalloc2::Operand>>,
    num_regs: usize,
    empty_block_set: Vec<regalloc2::Block>,
}

impl SsaMeta {
    fn new(ssa: &Ssa) -> Self {
        let num_insts = ssa.blocks.iter().map(|block| block.instrs.len()).sum();
        let num_blocks = ssa.blocks.len();
        let entry_block = regalloc2::Block::new(0);
        let block_instr_ranges =
            HashMap::from_iter(ssa.blocks.iter().enumerate().scan(0, |acc, (i, b)| {
                let begin_idx = regalloc2::Inst::new(*acc);
                let end_idx = regalloc2::Inst::new(*acc + b.instrs.len() - 1);
                *acc += b.instrs.len();
                Some((
                    regalloc2::Block::new(i),
                    regalloc2::InstRange::forward(begin_idx, end_idx),
                ))
            }));

        let mut block_succs: HashMap<usize, Vec<regalloc2::Block>> = HashMap::new();
        let mut block_preds: HashMap<usize, Vec<regalloc2::Block>> = HashMap::new();
        let mut add_succ_pred = |succ_idx, pred_idx| {
            let succ = regalloc2::Block::new(succ_idx);
            block_succs
                .entry(pred_idx)
                .and_modify(|succs| succs.push(succ))
                .or_insert(vec![succ]);

            let pred = regalloc2::Block::new(pred_idx);
            block_preds
                .entry(succ_idx)
                .and_modify(|preds| preds.push(pred))
                .or_insert(vec![pred]);
        };
        for (idx, block) in ssa.blocks.iter().enumerate() {
            match block.instrs.last() {
                Some(Instruction::Br(dst)) => {
                    add_succ_pred(*dst, idx);
                }
                Some(Instruction::Cbr(_, tr_dst, fa_dst)) => {
                    add_succ_pred(*tr_dst, idx);
                    add_succ_pred(*fa_dst, idx);
                }
                Some(Instruction::Ret(_)) => {}

                _ => unreachable!("Bad terminator for block."),
            }
        }

        let ret_set = HashSet::from_iter(
            ssa.blocks
                .iter()
                .flat_map(|b| b.instrs.iter())
                .enumerate()
                .filter_map(|(idx, instr)| matches!(instr, Instruction::Ret(_)).then_some(idx)),
        );
        let branch_set = HashSet::from_iter(
            ssa.blocks
                .iter()
                .flat_map(|b| b.instrs.iter())
                .enumerate()
                .filter_map(|(idx, instr)| {
                    matches!(instr, Instruction::Br(_) | Instruction::Cbr(..)).then_some(idx)
                }),
        );
        let new_use = |reg: &Register| {
            regalloc2::Operand::any_use(regalloc2::VReg::new(
                reg.0 as usize,
                regalloc2::RegClass::Int,
            ))
        };
        let new_def = |reg: &Register| {
            regalloc2::Operand::any_def(regalloc2::VReg::new(
                reg.0 as usize,
                regalloc2::RegClass::Int,
            ))
        };
        let moves = HashMap::from_iter(
            ssa.blocks
                .iter()
                .flat_map(|b| b.instrs.iter())
                .enumerate()
                .filter_map(|(idx, instr)| {
                    if let Instruction::Move(dst_reg, src_reg) = instr {
                        Some((idx, (new_use(src_reg), new_def(dst_reg))))
                    } else {
                        None
                    }
                }),
        );
        let inst_operands = HashMap::from_iter(
            ssa.blocks
                .iter()
                .flat_map(|b| b.instrs.iter())
                .enumerate()
                .map(|(idx, instr)| {
                    let opands = match instr {
                        Instruction::Add(a, b, c) => vec![new_def(a), new_use(b), new_use(c)],
                        Instruction::Br(_) => vec![],
                        Instruction::Cbr(a, _, _) => vec![new_use(a)],
                        Instruction::Move(a, b) => vec![new_def(a), new_use(b)],
                        Instruction::Movi(a, _) => vec![new_def(a)],
                        Instruction::Ret(a) => vec![new_use(a)],
                    };
                    (idx, opands)
                }),
        );
        let num_regs = ssa
            .blocks
            .iter()
            .rev()
            .flat_map(|b| b.instrs.iter().rev())
            .find_map(|i| {
                use std::cmp::max;
                match i {
                    Instruction::Add(a, b, c) => Some(max(a.0, max(b.0, c.0))),
                    Instruction::Cbr(a, _, _) => Some(a.0),
                    Instruction::Move(a, b) => Some(max(a.0, b.0)),
                    Instruction::Movi(a, _) => Some(a.0),
                    Instruction::Ret(a) => Some(a.0),
                    _ => None,
                }
            })
            .map(|r| (r + 1) as usize)
            .unwrap();

        SsaMeta {
            num_insts,
            num_blocks,
            entry_block,
            block_instr_ranges,
            block_succs,
            block_preds,
            dummy_block_params: Vec::new(),
            ret_set,
            branch_set,
            moves,
            inst_operands,
            num_regs,
            empty_block_set: Vec::new(),
        }
    }
}

impl regalloc2::Function for SsaMeta {
    fn num_insts(&self) -> usize {
        self.num_insts
    }

    fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    fn entry_block(&self) -> regalloc2::Block {
        self.entry_block
    }

    fn block_insns(&self, block: regalloc2::Block) -> regalloc2::InstRange {
        self.block_instr_ranges.get(&block).cloned().unwrap()
    }

    fn block_succs(&self, block: regalloc2::Block) -> &[regalloc2::Block] {
        self.block_succs
            .get(&block.index())
            .unwrap_or(&self.empty_block_set)
    }

    fn block_preds(&self, block: regalloc2::Block) -> &[regalloc2::Block] {
        self.block_preds
            .get(&block.index())
            .unwrap_or(&self.empty_block_set)
    }

    fn block_params(&self, _block: regalloc2::Block) -> &[regalloc2::VReg] {
        &self.dummy_block_params
    }

    fn is_ret(&self, insn: regalloc2::Inst) -> bool {
        self.ret_set.contains(&insn.index())
    }

    fn is_branch(&self, insn: regalloc2::Inst) -> bool {
        self.branch_set.contains(&insn.index())
    }

    fn branch_blockparams(
        &self,
        _block: regalloc2::Block,
        _insn: regalloc2::Inst,
        _succ_idx: usize,
    ) -> &[regalloc2::VReg] {
        &self.dummy_block_params
    }

    fn is_move(&self, insn: regalloc2::Inst) -> Option<(regalloc2::Operand, regalloc2::Operand)> {
        self.moves.get(&insn.index()).cloned()
    }

    fn inst_operands(&self, insn: regalloc2::Inst) -> &[regalloc2::Operand] {
        self.inst_operands.get(&insn.index()).unwrap()
    }

    fn inst_clobbers(&self, _insn: regalloc2::Inst) -> regalloc2::PRegSet {
        regalloc2::PRegSet::empty()
    }

    fn num_vregs(&self) -> usize {
        self.num_regs
    }

    fn spillslot_size(&self, _regclass: regalloc2::RegClass) -> usize {
        1
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
