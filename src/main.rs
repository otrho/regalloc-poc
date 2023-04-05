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

    let mut asm = AsmCompiler::compile(&ssa);
    println!("{asm}");
    let abstract_asm_result = asm.eval();

    let mut alloctr = AsmRegAllocator::new(&mut asm);
    if let Err(msg) = alloctr.reg_alloc() {
        println!("REG ALLOC FAILED: {msg}");
        return;
    }
    println!("{asm}");
    let allocd_asm_result = asm.eval();

    println!("EXPR RESULT: {expr_result}");
    println!("SSA RESULT: {ssa_result}");
    println!("ABSTRACT ASM RESULT: {abstract_asm_result}");
    println!("ALLOC'D ASM RESULT: {allocd_asm_result}");

    println!(
        "\n{}",
        if [
            expr_result,
            ssa_result,
            abstract_asm_result,
            allocd_asm_result
        ]
        .windows(2)
        .all(|results| results[0] == results[1])
        {
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
        Box::new(if depth > 8 {
            new_number()
        } else {
            if r < 20 {
                new_cond()
            } else if r < 60 {
                new_arith_op()
            } else {
                new_number()
            }
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
    num_regs: usize,
}

impl Display for Asm {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for (op_idx, op) in self.opcodes.iter().enumerate() {
            match op {
                Opcode::Label(idx) => write!(f, "{op_idx:>3} label_{idx}:\n")?,
                Opcode::Add(d, l, r) => write!(f, "{op_idx:>3}     add {d} {l} {r}\n")?,
                Opcode::Move(d, s) => write!(f, "{op_idx:>3}     mov {d} {s}\n")?,
                Opcode::Movi(d, i) => write!(f, "{op_idx:>3}     mov {d} {i}\n")?,
                Opcode::Jmp(idx) => write!(f, "{op_idx:>3}     jmp label_{idx}\n")?,
                Opcode::Jz(r, idx) => write!(f, "{op_idx:>3}     jz {r} label_{idx}\n")?,
                Opcode::Ret(r) => write!(f, "{op_idx:>3}     ret {r}\n")?,
                Opcode::Load(d, i) => write!(f, "{op_idx:>3}     load {d} {i}\n")?,
                Opcode::Store(i, s) => write!(f, "{op_idx:>3}     store {i} {s}\n")?,
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
enum Opcode {
    Label(LabelIdx),

    Add(Operand, Operand, Operand),
    Move(Operand, Operand),
    Movi(Operand, i64),

    Jmp(LabelIdx),
    Jz(Operand, LabelIdx),

    Ret(Operand),

    Load(Operand, Operand),  // reg, stack
    Store(Operand, Operand), // stack, reg
}

type LabelIdx = usize;

impl Opcode {
    fn rewrite_opcode_operand_register(&mut self, operand_idx: usize, reg_idx: usize) {
        self.rewrite_opcode_operand(operand_idx, Operand::Register(reg_idx))
    }

    fn _rewrite_opcode_operand_stack(&mut self, operand_idx: usize, slot_idx: usize) {
        self.rewrite_opcode_operand(operand_idx, Operand::Stack(slot_idx))
    }

    fn rewrite_opcode_operand(&mut self, operand_idx: usize, operand: Operand) {
        let opand_ref = match self {
            Opcode::Label(_) => unreachable!("Label has no operands"),
            Opcode::Add(a, b, c) => match operand_idx {
                0 => a,
                1 => b,
                2 => c,
                _ => unreachable!(),
            },
            Opcode::Move(a, b) => match operand_idx {
                0 => a,
                1 => b,
                _ => unreachable!(),
            },
            Opcode::Movi(a, _) => match operand_idx {
                0 => a,
                _ => unreachable!(),
            },
            Opcode::Jmp(_) => unreachable!("Jmp has no operands"),
            Opcode::Jz(a, _) => match operand_idx {
                0 => a,
                _ => unreachable!(),
            },
            Opcode::Ret(a) => match operand_idx {
                0 => a,
                _ => unreachable!(),
            },
            Opcode::Load(a, _) => match operand_idx {
                0 => a,
                _ => unreachable!(),
            },
            Opcode::Store(_, a) => match operand_idx {
                1 => a,
                _ => unreachable!(),
            },
        };
        *opand_ref = operand;
    }
}

#[derive(Clone, Copy, Debug)]
enum Operand {
    Register(usize),
    Stack(usize),
}

impl Display for Operand {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Operand::Register(r) => write!(f, "%{r}"),
            Operand::Stack(s) => write!(f, "[{s}]"),
        }
    }
}

impl Asm {
    fn eval(&self) -> i64 {
        // For now the ASM only ever jumps forward, so instead of having a PC and jumping around
        // arbitrarily we have a 'skip state' which skips instructions until a desired label is
        // found.
        let mut regs: HashMap<usize, i64> = HashMap::new();
        let mut stack: HashMap<usize, i64> = HashMap::new();

        macro_rules! greg {
            ($r: ident) => {
                regs.get($r).copied().unwrap()
            };
        }

        self.opcodes
            .iter()
            .fold((None, None), |(result, skip_idx), op| {
                if result.is_some() {
                    (result, None)
                } else {
                    if let Some(target_idx) = skip_idx {
                        match op {
                            Opcode::Label(label_idx) if label_idx == target_idx => (None, None),
                            _ => (None, skip_idx),
                        }
                    } else {
                        match op {
                            Opcode::Label(_) => (None, None),
                            Opcode::Add(
                                Operand::Register(d),
                                Operand::Register(l),
                                Operand::Register(r),
                            ) => {
                                let sum = greg!(l) + greg!(r);
                                regs.insert(*d, sum);
                                (None, None)
                            }
                            Opcode::Move(Operand::Register(d), Operand::Register(s)) => {
                                regs.insert(*d, greg!(s));
                                (None, None)
                            }
                            Opcode::Movi(Operand::Register(d), i) => {
                                regs.insert(*d, *i);
                                (None, None)
                            }
                            Opcode::Jmp(idx) => (None, Some(idx)),
                            Opcode::Jz(Operand::Register(c), idx) => {
                                if greg!(c) == 0 {
                                    (None, Some(idx))
                                } else {
                                    (None, None)
                                }
                            }
                            Opcode::Ret(Operand::Register(r)) => (Some(greg!(r)), None),
                            Opcode::Load(Operand::Register(d), Operand::Stack(slot)) => {
                                regs.insert(*d, stack.get(slot).copied().unwrap());
                                (None, None)
                            }
                            Opcode::Store(Operand::Stack(slot), Operand::Register(s)) => {
                                stack.insert(*slot, greg!(s));
                                (None, None)
                            }

                            _ => todo!("Eval {op:?}"),
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

    next_reg_idx: usize,
    reg_map: HashMap<ValueIdx, Operand>,

    next_label_idx: LabelIdx,
    label_map: HashMap<BlockIdx, LabelIdx>,
}

impl AsmCompiler {
    fn compile(ssa: &Ssa) -> Asm {
        AsmCompiler::new().compile_ssa(ssa)
    }

    fn new() -> Self {
        AsmCompiler {
            opcodes: Vec::new(),
            next_reg_idx: 1000,
            reg_map: HashMap::new(),
            next_label_idx: 0,
            label_map: HashMap::new(),
        }
    }

    fn compile_ssa(mut self, ssa: &Ssa) -> Asm {
        for (idx, block) in ssa.blocks.iter().enumerate() {
            self.compile_block(ssa, block, idx);
        }

        Asm {
            opcodes: self.opcodes,
            num_regs: self.next_reg_idx as usize,
        }
    }

    fn compile_block(&mut self, ssa: &Ssa, block: &Block, block_idx: BlockIdx) {
        let label_idx = self.label_idx_for_block(block_idx);
        self.opcodes.push(Opcode::Label(label_idx));

        for instr_val_idx in &block.instr_vals {
            let Value::Instruction(_block_idx, instr_idx) = ssa.values[*instr_val_idx] else {
                unreachable!("Block instruction value not an instruction.");
            };

            macro_rules! get_reg {
                ($val_idx: ident) => {{
                    self.compile_const(ssa, $val_idx);
                    self.reg_map.get(&$val_idx).copied().unwrap()
                }};
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
                    let l_reg = get_reg!(l_val_idx);
                    let r_reg = get_reg!(r_val_idx);
                    self.opcodes.push(Opcode::Add(instr_reg, l_reg, r_reg));
                }
                Instruction::Br(block_idx, block_arg) => {
                    if let (Some(block_arg_val_idx), Some(branch_arg_val_idx)) =
                        (ssa.blocks[block_idx].arg, block_arg)
                    {
                        let dst_arg_reg = get_reg_or_new!(block_arg_val_idx);
                        let src_arg_reg = get_reg!(branch_arg_val_idx);
                        self.opcodes.push(Opcode::Move(dst_arg_reg, src_arg_reg));
                    }
                    let block_label_idx = self.label_idx_for_block(block_idx);
                    self.opcodes.push(Opcode::Jmp(block_label_idx));
                }
                Instruction::Cbr(
                    c_val_idx,
                    tr_block_idx,
                    tr_block_arg,
                    fa_block_idx,
                    fa_block_arg,
                ) => {
                    if let (Some(block_arg_val_idx), Some(branch_arg_val_idx)) =
                        (ssa.blocks[fa_block_idx].arg, fa_block_arg)
                    {
                        let dst_arg_reg = get_reg_or_new!(block_arg_val_idx);
                        let src_arg_reg = get_reg!(branch_arg_val_idx);
                        self.opcodes.push(Opcode::Move(dst_arg_reg, src_arg_reg));
                    }
                    let fa_block_label_idx = self.label_idx_for_block(fa_block_idx);

                    let c_reg = get_reg!(c_val_idx);
                    self.opcodes.push(Opcode::Jz(c_reg, fa_block_label_idx));

                    if let (Some(block_arg_val_idx), Some(branch_arg_val_idx)) =
                        (ssa.blocks[tr_block_idx].arg, tr_block_arg)
                    {
                        let dst_arg_reg = get_reg_or_new!(block_arg_val_idx);
                        let src_arg_reg = get_reg!(branch_arg_val_idx);
                        self.opcodes.push(Opcode::Move(dst_arg_reg, src_arg_reg));
                    }
                    let tr_block_label_idx = self.label_idx_for_block(tr_block_idx);
                    self.opcodes.push(Opcode::Jmp(tr_block_label_idx));
                }
                Instruction::Ret(ret_val_idx) => {
                    let r_reg = get_reg!(ret_val_idx);
                    self.opcodes.push(Opcode::Ret(r_reg));
                }
            }
            self.reg_map.insert(*instr_val_idx, instr_reg);
        }
    }

    fn compile_const(&mut self, ssa: &Ssa, val_idx: ValueIdx) {
        if let Value::Const(n) = ssa.values[val_idx] {
            let const_reg = self.next_reg();
            self.opcodes.push(Opcode::Movi(const_reg, n));
            self.reg_map.insert(val_idx, const_reg);
        }
    }

    fn next_reg(&mut self) -> Operand {
        let reg_idx = self.next_reg_idx;
        self.next_reg_idx += 1;
        Operand::Register(reg_idx)
    }

    fn label_idx_for_block(&mut self, block_idx: BlockIdx) -> LabelIdx {
        *self.label_map.entry(block_idx).or_insert_with(|| {
            let label_idx = self.next_label_idx;
            self.next_label_idx += 1;
            label_idx
        })
    }
}

// }}}
// -------------------------------------------------------------------------------------------------
// AsmRegAllocator: Abstract Asm -> Allocated Asm Transformer {{{

struct AsmRegAllocator<'a> {
    asm: &'a mut Asm,
    block_bounds: Vec<regalloc2::InstRange>,
    entry_block: regalloc2::Block,
    block_succs: HashMap<usize, Vec<regalloc2::Block>>,
    block_preds: HashMap<usize, Vec<regalloc2::Block>>,
    dummy_block_params: Vec<regalloc2::VReg>,
    inst_operands: HashMap<usize, Vec<regalloc2::Operand>>,
    num_vregs: usize,
}

impl<'a> AsmRegAllocator<'a> {
    fn new(asm: &'a mut Asm) -> AsmRegAllocator {
        let label_offsets = asm
            .opcodes
            .iter()
            .chain(std::iter::once(&Opcode::Label(usize::MAX)))
            .enumerate()
            .filter_map(|(op_idx, op)| {
                if let Opcode::Label(label_idx) = op {
                    Some((op_idx, label_idx))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let block_bounds = label_offsets
            .windows(2)
            .map(|label_pairs| {
                regalloc2::InstRange::forward(
                    regalloc2::Inst::new(label_pairs[0].0),
                    regalloc2::Inst::new(label_pairs[1].0),
                )
            })
            .collect::<Vec<_>>();
        let label_map: HashMap<usize, usize> = HashMap::from_iter(
            (0..block_bounds.len())
                .zip(label_offsets)
                .map(|(block_idx, (_, label_idx))| (*label_idx, block_idx)),
        );

        let mut block_succs: HashMap<usize, Vec<regalloc2::Block>> = HashMap::new();
        let mut block_preds: HashMap<usize, Vec<regalloc2::Block>> = HashMap::new();
        block_succs.insert(block_bounds.len() - 1, Vec::new());
        block_preds.insert(0, Vec::new());

        let mut inst_operands: HashMap<usize, Vec<regalloc2::Operand>> = HashMap::new();

        let mut cur_block_idx = 0_usize;
        for (op_idx, op) in asm.opcodes.iter().enumerate() {
            match op {
                Opcode::Label(idx) => {
                    inst_operands.insert(op_idx, Vec::new());
                    cur_block_idx = *label_map.get(idx).unwrap();
                }
                Opcode::Add(a, b, c) => {
                    inst_operands.insert(
                        op_idx,
                        vec![Self::new_def(a), Self::new_use(b), Self::new_use(c)],
                    );
                }
                Opcode::Move(a, b) => {
                    inst_operands.insert(op_idx, vec![Self::new_def(a), Self::new_use(b)]);
                }
                Opcode::Movi(a, _) => {
                    inst_operands.insert(op_idx, vec![Self::new_def(a)]);
                }
                Opcode::Jmp(label_idx) => {
                    inst_operands.insert(op_idx, Vec::new());

                    let dst_block_idx = label_map.get(label_idx).copied().unwrap();
                    let dst_block = regalloc2::Block::new(dst_block_idx);
                    block_succs
                        .entry(cur_block_idx)
                        .and_modify(|v| v.push(dst_block))
                        .or_insert(vec![dst_block]);
                    let cur_block = regalloc2::Block::new(cur_block_idx);
                    block_preds
                        .entry(dst_block_idx)
                        .and_modify(|v| v.push(cur_block))
                        .or_insert(vec![cur_block]);
                }
                Opcode::Jz(a, label_idx) => {
                    inst_operands.insert(op_idx, vec![Self::new_use(a)]);

                    let dst_block_idx = label_map.get(label_idx).copied().unwrap();
                    let dst_block = regalloc2::Block::new(dst_block_idx);
                    block_succs
                        .entry(cur_block_idx)
                        .and_modify(|v| v.push(dst_block))
                        .or_insert(vec![dst_block]);
                    let cur_block = regalloc2::Block::new(cur_block_idx);
                    block_preds
                        .entry(dst_block_idx)
                        .and_modify(|v| v.push(cur_block))
                        .or_insert(vec![cur_block]);
                }
                Opcode::Ret(r) => {
                    inst_operands.insert(op_idx, vec![Self::new_use(r)]);
                }
                Opcode::Load(d, _) => {
                    inst_operands.insert(op_idx, vec![Self::new_def(d)]);
                }
                Opcode::Store(_, s) => {
                    inst_operands.insert(op_idx, vec![Self::new_use(s)]);
                }
            }
        }

        let entry_block = regalloc2::Block::new(0);
        let num_vregs = asm.num_regs;

        AsmRegAllocator {
            asm,
            block_bounds,
            entry_block,
            block_succs,
            block_preds,
            dummy_block_params: Vec::new(),
            inst_operands,
            num_vregs,
        }
    }

    fn reg_alloc(&mut self) -> Result<(), String> {
        let env = regalloc2::MachineEnv {
            preferred_regs_by_class: [
                (16..20)
                    .map(|i| regalloc2::PReg::new(i, regalloc2::RegClass::Int))
                    .collect(),
                Vec::new(),
            ],
            non_preferred_regs_by_class: [Vec::new(), Vec::new()],
            fixed_stack_slots: Vec::new(),
        };

        let opts = regalloc2::RegallocOptions {
            verbose_log: true,
            validate_ssa: false,
        };

        let alloc_output = regalloc2::run(self, &env, &opts).map_err(|e| format!("{e}"))?;
        let new_opcodes = (0..self.block_bounds.len())
            .flat_map(|block_idx| {
                alloc_output
                    .block_insts_and_edits(self, regalloc2::Block::new(block_idx))
                    .filter_map(|edit_or_inst| self.allocate_opcode(&alloc_output, edit_or_inst))
            })
            .collect();

        self.asm.opcodes = new_opcodes;

        Ok(())
    }

    fn allocate_opcode(
        &self,
        alloc_output: &regalloc2::Output,
        edit_or_inst: regalloc2::InstOrEdit,
    ) -> Option<Opcode> {
        match edit_or_inst {
            regalloc2::InstOrEdit::Inst(inst) => {
                let allocs = alloc_output.inst_allocs(inst);
                allocs.iter().enumerate().fold(
                    Some(self.asm.opcodes[inst.index()].clone()),
                    |inst, (opand_idx, alloc)| {
                        inst.and_then(|mut inst| match alloc.kind() {
                            regalloc2::AllocationKind::Reg => {
                                inst.rewrite_opcode_operand_register(opand_idx, alloc.index());
                                Some(inst)
                            }
                            regalloc2::AllocationKind::None => None,
                            regalloc2::AllocationKind::Stack => {
                                unreachable!("We have no opcodes which have stack use/defs.")
                            }
                        })
                    },
                )
            }
            regalloc2::InstOrEdit::Edit(regalloc2::Edit::Move { from, to }) => {
                Some(if to.is_stack() && from.is_reg() {
                    Opcode::Store(Operand::Stack(to.index()), Operand::Register(from.index()))
                } else if to.is_reg() && from.is_stack() {
                    Opcode::Load(Operand::Register(to.index()), Operand::Stack(from.index()))
                } else if to.is_reg() && from.is_reg() {
                    Opcode::Move(
                        Operand::Register(to.index()),
                        Operand::Register(from.index()),
                    )
                } else {
                    unimplemented!("We can only spill to/from registers.")
                })
            }
        }
    }

    fn new_use(reg: &Operand) -> regalloc2::Operand {
        if let Operand::Register(r) = reg {
            regalloc2::Operand::reg_use(regalloc2::VReg::new(*r as usize, regalloc2::RegClass::Int))
        } else {
            unreachable!("Generated operands must be registers at this stage.")
        }
    }

    fn new_def(reg: &Operand) -> regalloc2::Operand {
        if let Operand::Register(r) = reg {
            regalloc2::Operand::reg_def(regalloc2::VReg::new(*r as usize, regalloc2::RegClass::Int))
        } else {
            unreachable!("Generated operands must be registers at this stage.")
        }
    }
}

impl<'a> regalloc2::Function for AsmRegAllocator<'a> {
    fn num_insts(&self) -> usize {
        self.asm.opcodes.len()
    }

    fn num_blocks(&self) -> usize {
        self.block_bounds.len()
    }

    fn entry_block(&self) -> regalloc2::Block {
        self.entry_block
    }

    fn block_insns(&self, block: regalloc2::Block) -> regalloc2::InstRange {
        self.block_bounds[block.index()]
    }

    fn block_succs(&self, block: regalloc2::Block) -> &[regalloc2::Block] {
        self.block_succs.get(&block.index()).unwrap()
    }

    fn block_preds(&self, block: regalloc2::Block) -> &[regalloc2::Block] {
        self.block_preds.get(&block.index()).unwrap()
    }

    fn block_params(&self, _block: regalloc2::Block) -> &[regalloc2::VReg] {
        &self.dummy_block_params
    }

    fn is_ret(&self, insn: regalloc2::Inst) -> bool {
        matches!(self.asm.opcodes[insn.index()], Opcode::Ret(_))
    }

    fn is_branch(&self, insn: regalloc2::Inst) -> bool {
        matches!(
            self.asm.opcodes[insn.index()],
            Opcode::Jmp(_) | Opcode::Jz(..)
        )
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
        if let Opcode::Move(dst_reg, src_reg) = self.asm.opcodes[insn.index()] {
            Some((Self::new_use(&src_reg), Self::new_def(&dst_reg)))
        } else {
            None
        }
    }

    fn inst_operands(&self, insn: regalloc2::Inst) -> &[regalloc2::Operand] {
        self.inst_operands.get(&insn.index()).unwrap()
    }

    fn inst_clobbers(&self, _insn: regalloc2::Inst) -> regalloc2::PRegSet {
        regalloc2::PRegSet::empty()
    }

    fn num_vregs(&self) -> usize {
        self.num_vregs
    }

    fn spillslot_size(&self, _regclass: regalloc2::RegClass) -> usize {
        1
    }
}

// }}}
// -------------------------------------------------------------------------------------------------
