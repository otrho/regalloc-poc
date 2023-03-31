use std::{
    collections::HashMap,
    fmt::{Display, Formatter},
};

// -------------------------------------------------------------------------------------------------

fn main() {
    let expr = gen_rand_expr();
    println!("{expr}\n");
    let expr_result = expr.eval();

    let ssa = IrCompiler::compile(&expr);
    println!("{ssa}\n");
    let ssa_result = ssa.eval();

    let asm = AsmCompiler::compile(&ssa);
    println!("{asm}");
    let abstract_asm_result = asm.eval();

    //    if let Err(msg) = ssa.reg_alloc() {
    //        println!("REG ALLOC FAILED: {msg}");
    //        return;
    //    }
    //
    //    let asm_result = ssa.eval();
    //    println!("ASM RESULT: {asm_result}");

    println!("EXPR RESULT: {expr_result}");
    println!("SSA RESULT: {ssa_result}");
    println!("ABSTRACT ASM RESULT: {abstract_asm_result}");

    println!(
        "\n{}",
        if expr_result == ssa_result && ssa_result == abstract_asm_result {
            "SUCCESS"
        } else {
            "FAILURE"
        }
    );
}

// -------------------------------------------------------------------------------------------------
// Expr: Simple expression. {{{

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

// }}}
// -------------------------------------------------------------------------------------------------
// Ssa: Intermediate representation.  {{{

#[derive(Debug)]
struct Ssa {
    values: Vec<Value>,
    instrs: Vec<Instruction>,
    blocks: Vec<Block>,
}

impl Display for Ssa {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

#[derive(Debug)]
struct Block {
    arg: Option<ValueIdx>,
    instr_vals: Vec<ValueIdx>,
}

type BlockIdx = usize;

#[derive(Debug)]
enum Instruction {
    Add(ValueIdx, ValueIdx),
    Br(BlockIdx, Option<ValueIdx>),
    Cbr(
        ValueIdx,
        BlockIdx,
        Option<ValueIdx>,
        BlockIdx,
        Option<ValueIdx>,
    ),
    Ret(ValueIdx),
}

type InstructionIdx = usize;

#[derive(Debug)]
enum Value {
    Instruction(BlockIdx, InstructionIdx),
    Argument(BlockIdx),
    Const(i64),
}

type ValueIdx = usize;

impl Ssa {
    fn to_string(&self) -> String {
        let val_str = |val_idx: ValueIdx| {
            if let Value::Const(n) = self.values[val_idx] {
                format!("{n}")
            } else {
                format!("%{val_idx}")
            }
        };

        self.blocks
            .iter()
            .enumerate()
            .map(|(block_idx, block)| {
                let instrs_str = block
                    .instr_vals
                    .iter()
                    .map(|instr_val_idx| {
                        if let Value::Instruction(_block_idx, instr_idx) =
                            self.values[*instr_val_idx]
                        {
                            format!(
                                "    {}",
                                match self.instrs[instr_idx] {
                                    Instruction::Add(l_idx, r_idx) => format!(
                                        "%{instr_val_idx} = add {} {}",
                                        val_str(l_idx),
                                        val_str(r_idx)
                                    ),
                                    Instruction::Br(block_idx, arg_idx) => format!(
                                        "br block{block_idx}({})",
                                        arg_idx.map(|i| val_str(i)).unwrap_or(String::new())
                                    ),
                                    Instruction::Cbr(
                                        cond_val_idx,
                                        t_block_idx,
                                        t_arg_idx,
                                        f_block_idx,
                                        f_arg_idx,
                                    ) => format!(
                                        "cbr {} \
                                         block{t_block_idx}({}) \
                                         block{f_block_idx}({})",
                                        val_str(cond_val_idx),
                                        t_arg_idx.map(|i| val_str(i)).unwrap_or(String::new()),
                                        f_arg_idx.map(|i| val_str(i)).unwrap_or(String::new()),
                                    ),
                                    Instruction::Ret(ret_val_idx) =>
                                        format!("ret {}", val_str(ret_val_idx)),
                                }
                            )
                        } else {
                            unreachable!("Non instruction value in instruction list.")
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                format!(
                    "block{block_idx}({}):\n{}",
                    block
                        .arg
                        .map(|arg_val_idx| format!("%{arg_val_idx}"))
                        .unwrap_or(String::new()),
                    instrs_str
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    fn eval(&self) -> i64 {
        let mut vals = HashMap::new();
        self.eval_block(&mut vals, 0, None)
    }

    fn eval_block(
        &self,
        vals: &mut HashMap<ValueIdx, i64>,
        block_idx: BlockIdx,
        arg_val: Option<i64>,
    ) -> i64 {
        if let Some(arg_val_idx) = self.blocks[block_idx].arg {
            vals.insert(arg_val_idx, arg_val.unwrap());
        }

        fn resolve_val(
            ssa_vals: &[Value],
            vals: &HashMap<ValueIdx, i64>,
            val_idx: ValueIdx,
        ) -> i64 {
            vals.get(&val_idx).cloned().unwrap_or_else(|| {
                if let Value::Const(n) = ssa_vals[val_idx] {
                    n
                } else {
                    unreachable!("Unable to resolve value %{val_idx}.")
                }
            })
        }

        macro_rules! resolve_val {
            ($val_idx: ident) => {
                resolve_val(&self.values, vals, $val_idx)
            };
        }

        self.blocks[block_idx]
            .instr_vals
            .iter()
            .map(|instr_val_idx| {
                if let Value::Instruction(_block_idx, instr_idx) = self.values[*instr_val_idx] {
                    match self.instrs[instr_idx] {
                        Instruction::Add(l_val_idx, r_val_idx) => {
                            let result = resolve_val!(l_val_idx) + resolve_val!(r_val_idx);
                            vals.insert(*instr_val_idx, result);
                            result
                        }
                        Instruction::Br(block_idx, arg_val_idx) => {
                            self.eval_block(vals, block_idx, arg_val_idx.map(|i| resolve_val!(i)))
                        }
                        Instruction::Cbr(
                            c_val_idx,
                            tr_block_idx,
                            tr_arg_val_idx,
                            fa_block_idx,
                            fa_arg_val_idx,
                        ) => {
                            if resolve_val!(c_val_idx) != 0 {
                                self.eval_block(
                                    vals,
                                    tr_block_idx,
                                    tr_arg_val_idx.map(|i| resolve_val!(i)),
                                )
                            } else {
                                self.eval_block(
                                    vals,
                                    fa_block_idx,
                                    fa_arg_val_idx.map(|i| resolve_val!(i)),
                                )
                            }
                        }
                        Instruction::Ret(ret_val_idx) => resolve_val!(ret_val_idx),
                    }
                } else {
                    unreachable!("Non instruction value in instruction list.")
                }
            })
            .last()
            .unwrap()
    }
}

// }}}
// -------------------------------------------------------------------------------------------------
// IrCompiler: Expr -> Ssa Compiler {{{

struct IrCompiler {
    values: Vec<Value>,
    instrs: Vec<Instruction>,
    blocks: Vec<Block>,

    cur_block_idx: usize,
}

// -------------------------------------------------------------------------------------------------

impl IrCompiler {
    fn compile(expr: &Expr) -> Ssa {
        let mut compiler = IrCompiler::new();

        let result_val = compiler.compile_expr(expr);
        compiler.append_instr(Instruction::Ret(result_val));

        Ssa {
            values: compiler.values,
            instrs: compiler.instrs,
            blocks: compiler.blocks,
        }
    }

    fn new() -> Self {
        let blocks = vec![Block {
            arg: None,
            instr_vals: Vec::new(),
        }];
        IrCompiler {
            blocks,
            values: Vec::new(),
            instrs: Vec::new(),
            cur_block_idx: 0,
        }
    }

    fn compile_expr(&mut self, expr: &Expr) -> ValueIdx {
        match expr {
            Expr::Number(n) => {
                let val_idx = self.values.len();
                self.values.push(Value::Const(*n));
                val_idx
            }
            Expr::Add(l, r) => {
                let l_val_idx = self.compile_expr(l);
                let r_val_idx = self.compile_expr(r);
                self.append_instr(Instruction::Add(l_val_idx, r_val_idx))
            }
            Expr::Cond(c, t, f) => {
                let c_val_idx = self.compile_expr(c);
                let c_end_block_idx = self.cur_block_idx;

                let t_begin_block_idx = self.new_block();
                let t_val_idx = self.compile_expr(t);
                let t_end_block_idx = self.cur_block_idx;

                let f_begin_block_idx = self.new_block();
                let f_val_idx = self.compile_expr(f);
                let f_end_block_idx = self.cur_block_idx;

                self.append_instr_to(
                    c_end_block_idx,
                    Instruction::Cbr(c_val_idx, t_begin_block_idx, None, f_begin_block_idx, None),
                );

                let (j_block_idx, j_block_arg_idx) = self.new_block_with_arg();
                self.append_instr_to(
                    t_end_block_idx,
                    Instruction::Br(j_block_idx, Some(t_val_idx)),
                );
                self.append_instr_to(
                    f_end_block_idx,
                    Instruction::Br(j_block_idx, Some(f_val_idx)),
                );
                j_block_arg_idx
            }
        }
    }

    fn new_block(&mut self) -> BlockIdx {
        let idx = self.blocks.len();
        self.blocks.push(Block {
            instr_vals: Vec::new(),
            arg: None,
        });
        self.cur_block_idx = idx;
        idx
    }

    fn new_block_with_arg(&mut self) -> (BlockIdx, ValueIdx) {
        let block_idx = self.blocks.len();
        let arg_val_idx = self.values.len();
        self.values.push(Value::Argument(block_idx));
        self.blocks.push(Block {
            instr_vals: Vec::new(),
            arg: Some(arg_val_idx),
        });
        self.cur_block_idx = block_idx;
        (block_idx, arg_val_idx)
    }

    fn append_instr(&mut self, instr: Instruction) -> ValueIdx {
        self.append_instr_to(self.cur_block_idx, instr)
    }

    fn append_instr_to(&mut self, block_idx: usize, instr: Instruction) -> ValueIdx {
        let instr_idx = self.instrs.len();
        self.instrs.push(instr);

        let instr_val_idx = self.values.len();
        self.values.push(Value::Instruction(block_idx, instr_idx));

        self.blocks[block_idx].instr_vals.push(instr_val_idx);
        instr_val_idx
    }
}

// }}}
// -------------------------------------------------------------------------------------------------
// Asm: Assembly code {{{

struct Asm {
    opcodes: Vec<Opcode>,
}

impl Display for Asm {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for op in &self.opcodes {
            match op {
                Opcode::BlockBegin(idx) => write!(f, "block_begin_{idx}\n")?,
                Opcode::BlockEnd(idx) => write!(f, "block_end_{idx}\n\n")?,
                Opcode::Add(d, l, r) => write!(f, "    add {d} {l} {r}\n")?,
                Opcode::Move(d, s) => write!(f, "    mov {d} {s}\n")?,
                Opcode::Movi(d, i) => write!(f, "    mov {d} {i}\n")?,
                Opcode::JmpBlockBegin(idx) => write!(f, "    jmp block_begin_{idx}\n")?,
                Opcode::JmpBlockEnd(idx) => write!(f, "    jmp block_end_{idx}\n")?,
                Opcode::JzBlockBegin(r, idx) => write!(f, "    jz {r} block_begin_{idx}\n")?,
                Opcode::Ret(r) => write!(f, "    ret {r}\n")?,
            }
        }
        Ok(())
    }
}

enum Opcode {
    BlockBegin(BlockIdx),
    BlockEnd(BlockIdx),

    Add(Register, Register, Register),
    Move(Register, Register),
    Movi(Register, i64),

    JmpBlockBegin(BlockIdx),
    JmpBlockEnd(BlockIdx),
    JzBlockBegin(Register, BlockIdx),

    Ret(Register),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Register(u64);

impl Display for Register {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.0)
    }
}

impl Asm {
    fn eval(&self) -> i64 {
        // For now the ASM only ever jumps forward, so instead of having a PC and jumping around
        // arbitrarily we have a 'skip state' which skips instructions until a desired label is
        // found.
        let mut regs: HashMap<Register, i64> = HashMap::new();
        self.opcodes
            .iter()
            .fold((None, None), |(result, skip_idx), op| {
                if result.is_some() {
                    (result, None)
                } else {
                    if let Some((target_idx, begin)) = skip_idx {
                        match op {
                            Opcode::BlockBegin(block_idx) if block_idx == target_idx && begin => {
                                (None, None)
                            }
                            Opcode::BlockEnd(block_idx) if block_idx == target_idx && !begin => {
                                (None, None)
                            }
                            _ => (None, skip_idx),
                        }
                    } else {
                        match op {
                            Opcode::BlockBegin(_) => (None, None),
                            Opcode::BlockEnd(_) => (None, None),
                            Opcode::Add(d, l, r) => {
                                let sum = regs.get(l).unwrap() + regs.get(r).unwrap();
                                regs.insert(*d, sum);
                                (None, None)
                            }
                            Opcode::Move(d, s) => {
                                regs.insert(*d, regs.get(s).copied().unwrap());
                                (None, None)
                            }
                            Opcode::Movi(d, i) => {
                                regs.insert(*d, *i);
                                (None, None)
                            }
                            Opcode::JmpBlockBegin(idx) => (None, Some((idx, true))),
                            Opcode::JmpBlockEnd(idx) => (None, Some((idx, false))),
                            Opcode::JzBlockBegin(c, idx) => {
                                if regs.get(c).copied().unwrap() == 0 {
                                    (None, Some((idx, true)))
                                } else {
                                    (None, None)
                                }
                            }
                            Opcode::Ret(ret_reg) => (regs.get(ret_reg).copied(), None),
                        }
                    }
                }
            })
            .0
            .unwrap()
    }
}

// }}}
// -------------------------------------------------------------------------------------------------
// AsmCompiler: Ssa -> Asm Compiler {{{

struct AsmCompiler {
    opcodes: Vec<Opcode>,

    next_reg_idx: u64,
    reg_map: HashMap<ValueIdx, Register>,
}

impl AsmCompiler {
    fn compile(ssa: &Ssa) -> Asm {
        AsmCompiler::new().compile_ssa(ssa)
    }

    fn new() -> Self {
        AsmCompiler {
            opcodes: Vec::new(),
            next_reg_idx: 0,
            reg_map: HashMap::new(),
        }
    }

    fn compile_ssa(mut self, ssa: &Ssa) -> Asm {
        for (idx, block) in ssa.blocks.iter().enumerate() {
            self.compile_block(ssa, block, idx);
        }

        Asm {
            opcodes: self.opcodes,
        }
    }

    fn compile_block(&mut self, ssa: &Ssa, block: &Block, block_idx: BlockIdx) {
        self.opcodes.push(Opcode::BlockBegin(block_idx));
        for instr_val_idx in &block.instr_vals {
            let Value::Instruction(_block_idx, instr_idx) = ssa.values[*instr_val_idx] else {
                unreachable!("Block instruction value not an instruction.");
            };

            macro_rules! get_reg {
                ($val_idx: ident) => {
                    self.reg_map.get(&$val_idx).copied().unwrap()
                };
            }
            macro_rules! get_reg_or_new {
                ($val_idx: ident) => {
                    if self.reg_map.contains_key(&$val_idx) {
                        self.reg_map.get(&$val_idx).copied().unwrap()
                    } else {
                        let reg = self.next_reg();
                        self.reg_map.insert($val_idx, reg);
                        reg
                    }
                };
            }

            let instr_reg = self.next_reg();
            match ssa.instrs[instr_idx] {
                Instruction::Add(l_val_idx, r_val_idx) => {
                    self.compile_const(ssa, l_val_idx);
                    self.compile_const(ssa, r_val_idx);
                    self.opcodes.push(Opcode::Add(
                        instr_reg,
                        get_reg!(l_val_idx),
                        get_reg!(r_val_idx),
                    ));
                }
                Instruction::Br(block_idx, block_arg) => {
                    if let (Some(block_arg_val_idx), Some(branch_arg_val_idx)) =
                        (ssa.blocks[block_idx].arg, block_arg)
                    {
                        let dst_arg_reg = get_reg_or_new!(block_arg_val_idx);
                        let src_arg_reg = get_reg!(branch_arg_val_idx);
                        self.opcodes.push(Opcode::Move(dst_arg_reg, src_arg_reg));
                    }
                    self.opcodes.push(Opcode::JmpBlockBegin(block_idx));
                }
                Instruction::Cbr(
                    c_val_idx,
                    tr_block_idx,
                    tr_block_arg,
                    fa_block_idx,
                    fa_block_arg,
                ) => {
                    self.compile_const(ssa, c_val_idx);

                    if let (Some(block_arg_val_idx), Some(branch_arg_val_idx)) =
                        (ssa.blocks[fa_block_idx].arg, fa_block_arg)
                    {
                        let dst_arg_reg = get_reg_or_new!(block_arg_val_idx);
                        let src_arg_reg = get_reg!(branch_arg_val_idx);
                        self.opcodes.push(Opcode::Move(dst_arg_reg, src_arg_reg));
                    }
                    self.opcodes
                        .push(Opcode::JzBlockBegin(get_reg!(c_val_idx), fa_block_idx));

                    if let (Some(block_arg_val_idx), Some(branch_arg_val_idx)) =
                        (ssa.blocks[tr_block_idx].arg, tr_block_arg)
                    {
                        let dst_arg_reg = get_reg_or_new!(block_arg_val_idx);
                        let src_arg_reg = get_reg!(branch_arg_val_idx);
                        self.opcodes.push(Opcode::Move(dst_arg_reg, src_arg_reg));
                    }
                    self.compile_block(ssa, &ssa.blocks[tr_block_idx], tr_block_idx);
                    self.opcodes.push(Opcode::JmpBlockEnd(fa_block_idx));

                    self.compile_block(ssa, &ssa.blocks[fa_block_idx], fa_block_idx);
                }
                Instruction::Ret(ret_val_idx) => {
                    self.opcodes.push(Opcode::Ret(get_reg!(ret_val_idx)));
                }
            }
            self.reg_map.insert(*instr_val_idx, instr_reg);
        }
        self.opcodes.push(Opcode::BlockEnd(block_idx));
    }

    fn compile_const(&mut self, ssa: &Ssa, val_idx: ValueIdx) {
        if let Value::Const(n) = ssa.values[val_idx] {
            let const_reg = self.next_reg();
            self.opcodes.push(Opcode::Movi(const_reg, n));
            self.reg_map.insert(val_idx, const_reg);
        }
    }

    fn next_reg(&mut self) -> Register {
        let reg_idx = self.next_reg_idx;
        self.next_reg_idx += 1;
        Register(reg_idx)
    }
}

// }}}
// -------------------------------------------------------------------------------------------------
// First attempt at SSA reg alloc {{{

/*
{
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
*/
// }}}
// -------------------------------------------------------------------------------------------------
