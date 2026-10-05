	.def	@feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 0
	.file	"q1t.cpp"
	.def	ggml_cpu_q1t_supports_mul_mat;
	.scl	2;
	.type	32;
	.endef
	.text
	.globl	ggml_cpu_q1t_supports_mul_mat   # -- Begin function ggml_cpu_q1t_supports_mul_mat
	.p2align	4
ggml_cpu_q1t_supports_mul_mat:          # @ggml_cpu_q1t_supports_mul_mat
.seh_proc ggml_cpu_q1t_supports_mul_mat
# %bb.0:
	pushq	%rsi
	.seh_pushreg %rsi
	pushq	%rdi
	.seh_pushreg %rdi
	pushq	%rbx
	.seh_pushreg %rbx
	subq	$32, %rsp
	.seh_stackalloc 32
	.seh_endprologue
	movq	152(%rcx), %rsi
	cmpl	$41, (%rsi)
	jne	.LBB0_6
# %bb.1:
	movq	%rcx, %rdi
	movq	160(%rcx), %rbx
	movl	$41, %ecx
	callq	ggml_get_type_traits_cpu
	cmpl	$8, 16(%rax)
	jne	.LBB0_6
# %bb.2:
	callq	ggml_cpu_has_avx2
	testl	%eax, %eax
	je	.LBB0_6
# %bb.3:
	movl	"?$TSS0@?1??ggml_cpu_q1t_supports_mul_mat@@9@4HA"(%rip), %eax
	movl	_tls_index(%rip), %ecx
	movq	%gs:88, %rdx
	movq	(%rdx,%rcx,8), %rcx
	cmpl	_Init_thread_epoch@SECREL32(%rcx), %eax
	jg	.LBB0_29
.LBB0_4:
	cmpb	$0, "?disabled@?1??ggml_cpu_q1t_supports_mul_mat@@9@4_NB"(%rip)
	jne	.LBB0_6
# %bb.9:
	cmpl	$0, (%rbx)
	jne	.LBB0_6
# %bb.11:
	testb	$127, 16(%rsi)
	jne	.LBB0_6
# %bb.13:
	cmpq	$2, 24(%rsi)
	jl	.LBB0_6
# %bb.15:
	cmpq	$4, 24(%rbx)
	jl	.LBB0_6
# %bb.17:
	cmpq	$1, 32(%rsi)
	jne	.LBB0_6
# %bb.18:
	cmpq	$1, 40(%rsi)
	jne	.LBB0_6
# %bb.19:
	cmpq	$1, 40(%rbx)
	jne	.LBB0_6
# %bb.20:
	movq	%rsi, %rcx
	callq	ggml_is_contiguous
	testb	%al, %al
	je	.LBB0_6
# %bb.21:
	cmpl	$0, (%rdi)
	je	.LBB0_27
.LBB0_6:
	xorl	%eax, %eax
.LBB0_28:
                                        # kill: def $al killed $al killed $eax
	addq	$32, %rsp
	popq	%rbx
	popq	%rdi
	popq	%rsi
	retq
.LBB0_27:
	cmpq	$4, 48(%rdi)
	sete	%al
	jmp	.LBB0_28
.LBB0_29:
	leaq	"?$TSS0@?1??ggml_cpu_q1t_supports_mul_mat@@9@4HA"(%rip), %rcx
	callq	_Init_thread_header
	cmpl	$-1, "?$TSS0@?1??ggml_cpu_q1t_supports_mul_mat@@9@4HA"(%rip)
	jne	.LBB0_4
# %bb.30:
	leaq	"??_C@_0BA@PNIHBCGC@GGML_NO_Q1_TILE?$AA@"(%rip), %rcx
	callq	getenv
	testq	%rax, %rax
	setne	"?disabled@?1??ggml_cpu_q1t_supports_mul_mat@@9@4_NB"(%rip)
	leaq	"?$TSS0@?1??ggml_cpu_q1t_supports_mul_mat@@9@4HA"(%rip), %rcx
	callq	_Init_thread_footer
	jmp	.LBB0_4
	.seh_endproc
                                        # -- End function
	.def	ggml_compute_forward_mul_mat_q1t;
	.scl	2;
	.type	32;
	.endef
	.globl	ggml_compute_forward_mul_mat_q1t # -- Begin function ggml_compute_forward_mul_mat_q1t
	.p2align	4
ggml_compute_forward_mul_mat_q1t:       # @ggml_compute_forward_mul_mat_q1t
.seh_proc ggml_compute_forward_mul_mat_q1t
# %bb.0:
	pushq	%r15
	.seh_pushreg %r15
	pushq	%r14
	.seh_pushreg %r14
	pushq	%r13
	.seh_pushreg %r13
	pushq	%r12
	.seh_pushreg %r12
	pushq	%rsi
	.seh_pushreg %rsi
	pushq	%rdi
	.seh_pushreg %rdi
	pushq	%rbp
	.seh_pushreg %rbp
	pushq	%rbx
	.seh_pushreg %rbx
	subq	$536, %rsp                      # imm = 0x218
	.seh_stackalloc 536
	.seh_endprologue
	movq	%rcx, 136(%rsp)                 # 8-byte Spill
	movq	152(%rdx), %rsi
	movq	%rdx, 168(%rsp)                 # 8-byte Spill
	movq	160(%rdx), %rax
	testq	%rsi, %rsi
	je	.LBB1_1
# %bb.2:
	movl	16(%rsi), %edi
	movq	24(%rsi), %rcx
	movq	%rcx, 104(%rsp)                 # 8-byte Spill
	movq	56(%rsi), %rcx
	movq	%rcx, 160(%rsp)                 # 8-byte Spill
	testq	%rax, %rax
	je	.LBB1_4
.LBB1_5:
	movq	16(%rax), %rdx
	movq	24(%rax), %rbx
	movq	32(%rax), %rax
	movq	%rax, 120(%rsp)                 # 8-byte Spill
	jmp	.LBB1_6
.LBB1_1:
	movq	$0, 104(%rsp)                   # 8-byte Folded Spill
	xorl	%edi, %edi
	movq	$0, 160(%rsp)                   # 8-byte Folded Spill
	testq	%rax, %rax
	jne	.LBB1_5
.LBB1_4:
	xorl	%ebx, %ebx
	xorl	%edx, %edx
	movq	$0, 120(%rsp)                   # 8-byte Folded Spill
.LBB1_6:
	movq	168(%rsp), %rax                 # 8-byte Reload
	movq	48(%rax), %rcx
	movq	%rcx, 112(%rsp)                 # 8-byte Spill
	movq	56(%rax), %r13
	movq	64(%rax), %rax
	movq	%rax, 272(%rsp)                 # 8-byte Spill
	movq	136(%rsp), %rax                 # 8-byte Reload
	movslq	(%rax), %rcx
	movq	%rcx, 152(%rsp)                 # 8-byte Spill
	movslq	4(%rax), %r14
	movl	$8, %ecx
	callq	ggml_row_size
	movq	%rax, %r15
	movq	104(%rsp), %rax                 # 8-byte Reload
	movq	%rax, %r12
	shrq	$63, %r12
	addq	%rax, %r12
	sarq	%r12
	movl	$41, %ecx
	callq	ggml_get_type_traits_cpu
	movq	8(%rax), %rbp
	callq	ggml_is_numa
                                        # kill: def $al killed $al def $rax
	xorb	$1, %al
	leal	(%rax,%rax), %ecx
	movq	%r14, 184(%rsp)                 # 8-byte Spill
	movq	%r14, %r8
                                        # kill: def $cl killed $cl killed $ecx
	shlq	%cl, %r8
	leaq	-1(%r12), %rcx
	movq	%r12, 64(%rsp)                  # 8-byte Spill
	leaq	(%r12,%r8), %rax
	decq	%rax
	movq	%rax, %rdx
	orq	%r8, %rdx
	shrq	$32, %rdx
	je	.LBB1_7
# %bb.8:
	cqto
	idivq	%r8
	jmp	.LBB1_9
.LBB1_7:
                                        # kill: def $eax killed $eax killed $rax
	xorl	%edx, %edx
	divl	%r8d
                                        # kill: def $eax killed $eax def $rax
.LBB1_9:
	cmpq	$2, %rax
	movl	$1, %r8d
	cmovgeq	%rax, %r8
	addq	%r8, %rcx
	movq	%rcx, %rax
	orq	%r8, %rax
	shrq	$32, %rax
	je	.LBB1_10
# %bb.11:
	movq	%rcx, %rax
	cqto
	idivq	%r8
	jmp	.LBB1_12
.LBB1_10:
	movl	%ecx, %eax
	xorl	%edx, %edx
	divl	%r8d
                                        # kill: def $eax killed $eax def $rax
.LBB1_12:
	movq	%r15, %rcx
	imulq	%rbx, %rcx
	movq	%rcx, 264(%rsp)                 # 8-byte Spill
	cmpq	152(%rsp), %rax                 # 8-byte Folded Reload
	movq	%rsi, 192(%rsp)                 # 8-byte Spill
	movl	%edi, 84(%rsp)                  # 4-byte Spill
	movq	%rbx, 128(%rsp)                 # 8-byte Spill
	movq	%r13, 280(%rsp)                 # 8-byte Spill
	movq	%r15, 176(%rsp)                 # 8-byte Spill
	jle	.LBB1_26
# %bb.13:
	leaq	(%r15,%r15,2), %rcx
	movq	%rcx, 528(%rsp)                 # 8-byte Spill
	leaq	(,%r15,4), %rcx
	movq	%rcx, 520(%rsp)                 # 8-byte Spill
	leaq	(,%r13,2), %rcx
	addq	%r13, %rcx
	movq	%rcx, 432(%rsp)                 # 8-byte Spill
	leaq	(%r8,%r8), %rcx
	movq	%rcx, 424(%rsp)                 # 8-byte Spill
	movq	112(%rsp), %rcx                 # 8-byte Reload
	leaq	(%rcx,%rcx), %rdx
	movq	%rdx, 456(%rsp)                 # 8-byte Spill
	leaq	(,%r13,4), %rdx
	movq	%rdx, 512(%rsp)                 # 8-byte Spill
	imulq	%r8, %rcx
	addq	%rcx, %rcx
	movq	%rcx, 416(%rsp)                 # 8-byte Spill
	leaq	(%r15,%r15), %rcx
	movq	%rcx, 504(%rsp)                 # 8-byte Spill
	leaq	(,%r13,2), %rcx
	movq	%rcx, 408(%rsp)                 # 8-byte Spill
	movq	152(%rsp), %rcx                 # 8-byte Reload
	movq	%rbp, 480(%rsp)                 # 8-byte Spill
	movq	%r8, 448(%rsp)                  # 8-byte Spill
	movq	%rax, 440(%rsp)                 # 8-byte Spill
	jmp	.LBB1_14
	.p2align	4
.LBB1_25:                               #   in Loop: Header=BB1_14 Depth=1
	movq	136(%rsp), %rax                 # 8-byte Reload
	movq	24(%rax), %rcx
	movl	$1, %edx
	callq	ggml_threadpool_chunk_add
	movslq	%eax, %rcx
	movq	440(%rsp), %rax                 # 8-byte Reload
	cmpq	%rcx, %rax
	movq	448(%rsp), %r8                  # 8-byte Reload
	jle	.LBB1_26
.LBB1_14:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_16 Depth 2
                                        #       Child Loop BB1_18 Depth 3
                                        #         Child Loop BB1_38 Depth 4
                                        #         Child Loop BB1_22 Depth 4
	movq	%rcx, %r9
	imulq	%r8, %r9
	leaq	(%r9,%r8), %rdx
	movq	64(%rsp), %rax                  # 8-byte Reload
	cmpq	%rax, %rdx
	cmovgeq	%rax, %rdx
	movq	%rdx, 464(%rsp)                 # 8-byte Spill
	cmpq	%r9, %rax
	jle	.LBB1_25
# %bb.15:                               #   in Loop: Header=BB1_14 Depth=1
	movq	424(%rsp), %rdx                 # 8-byte Reload
	imulq	%rcx, %rdx
	incq	%rdx
	imulq	112(%rsp), %rdx                 # 8-byte Folded Reload
	movq	432(%rsp), %rax                 # 8-byte Reload
	leaq	(%rax,%rdx), %r8
	movq	%r8, 240(%rsp)                  # 8-byte Spill
	imulq	416(%rsp), %rcx                 # 8-byte Folded Reload
	addq	%rcx, %rax
	movq	%rax, 232(%rsp)                 # 8-byte Spill
	movq	408(%rsp), %rax                 # 8-byte Reload
	leaq	(%rax,%rdx), %r8
	movq	%r8, 224(%rsp)                  # 8-byte Spill
	addq	%rcx, %rax
	movq	%rax, 216(%rsp)                 # 8-byte Spill
	movq	%rdx, 248(%rsp)                 # 8-byte Spill
	leaq	(%r13,%rdx), %rax
	movq	%rax, 208(%rsp)                 # 8-byte Spill
	leaq	(%rcx,%r13), %rax
	movq	%rax, 200(%rsp)                 # 8-byte Spill
	jmp	.LBB1_16
	.p2align	4
.LBB1_24:                               #   in Loop: Header=BB1_16 Depth=2
	movq	472(%rsp), %r9                  # 8-byte Reload
	incq	%r9
	movq	456(%rsp), %rax                 # 8-byte Reload
	addq	%rax, 240(%rsp)                 # 8-byte Folded Spill
	addq	%rax, 232(%rsp)                 # 8-byte Folded Spill
	addq	%rax, 224(%rsp)                 # 8-byte Folded Spill
	addq	%rax, 216(%rsp)                 # 8-byte Folded Spill
	addq	%rax, 208(%rsp)                 # 8-byte Folded Spill
	addq	%rax, 200(%rsp)                 # 8-byte Folded Spill
	addq	%rax, 248(%rsp)                 # 8-byte Folded Spill
	movq	256(%rsp), %rcx                 # 8-byte Reload
	addq	%rax, %rcx
	cmpq	464(%rsp), %r9                  # 8-byte Folded Reload
	movq	192(%rsp), %rsi                 # 8-byte Reload
	jge	.LBB1_25
.LBB1_16:                               #   Parent Loop BB1_14 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB1_18 Depth 3
                                        #         Child Loop BB1_38 Depth 4
                                        #         Child Loop BB1_22 Depth 4
	movq	%rcx, 256(%rsp)                 # 8-byte Spill
	leaq	(%r9,%r9), %rax
	movq	248(%rsi), %rcx
	movq	160(%rsp), %r8                  # 8-byte Reload
	imulq	%r8, %rax
	addq	%rcx, %rax
	movq	%r9, 472(%rsp)                  # 8-byte Spill
	leaq	1(,%r9,2), %rdx
	imulq	%r8, %rdx
	addq	%rcx, %rdx
	movq	%rax, 352(%rsp)
	movq	%rdx, 360(%rsp)
	cmpq	$0, 120(%rsp)                   # 8-byte Folded Reload
	jle	.LBB1_24
# %bb.17:                               #   in Loop: Header=BB1_16 Depth=2
	movq	256(%rsp), %rcx                 # 8-byte Reload
	movq	248(%rsp), %rdx                 # 8-byte Reload
	movq	200(%rsp), %r8                  # 8-byte Reload
	movq	208(%rsp), %r9                  # 8-byte Reload
	movq	216(%rsp), %r10                 # 8-byte Reload
	movq	224(%rsp), %r11                 # 8-byte Reload
	movq	232(%rsp), %rsi                 # 8-byte Reload
	movq	240(%rsp), %rdi                 # 8-byte Reload
	xorl	%r14d, %r14d
	xorl	%r15d, %r15d
	jmp	.LBB1_18
	.p2align	4
.LBB1_23:                               #   in Loop: Header=BB1_18 Depth=3
	movq	488(%rsp), %r15                 # 8-byte Reload
	incq	%r15
	movq	288(%rsp), %r14                 # 8-byte Reload
	addq	264(%rsp), %r14                 # 8-byte Folded Reload
	movq	272(%rsp), %rax                 # 8-byte Reload
	movq	296(%rsp), %rdi                 # 8-byte Reload
	addq	%rax, %rdi
	movq	304(%rsp), %rsi                 # 8-byte Reload
	addq	%rax, %rsi
	movq	496(%rsp), %r11                 # 8-byte Reload
	addq	%rax, %r11
	movq	312(%rsp), %r10                 # 8-byte Reload
	addq	%rax, %r10
	movq	88(%rsp), %r9                   # 8-byte Reload
	addq	%rax, %r9
	movq	144(%rsp), %r8                  # 8-byte Reload
	addq	%rax, %r8
	addq	%rax, %rdx
	movq	72(%rsp), %rcx                  # 8-byte Reload
	addq	%rax, %rcx
	cmpq	120(%rsp), %r15                 # 8-byte Folded Reload
	movq	128(%rsp), %rbx                 # 8-byte Reload
	je	.LBB1_24
.LBB1_18:                               #   Parent Loop BB1_14 Depth=1
                                        #     Parent Loop BB1_16 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB1_38 Depth 4
                                        #         Child Loop BB1_22 Depth 4
	movq	%r15, 488(%rsp)                 # 8-byte Spill
	movq	%rdi, 296(%rsp)                 # 8-byte Spill
	movq	%rsi, 304(%rsp)                 # 8-byte Spill
	movq	136(%rsp), %rax                 # 8-byte Reload
	movq	16(%rax), %rdi
	movq	168(%rsp), %rax                 # 8-byte Reload
	movq	248(%rax), %r12
	cmpq	$4, %rbx
	movq	%rdx, 96(%rsp)                  # 8-byte Spill
	movq	%r8, 144(%rsp)                  # 8-byte Spill
	movq	%r9, 88(%rsp)                   # 8-byte Spill
	movq	%r10, 312(%rsp)                 # 8-byte Spill
	movq	%r14, 288(%rsp)                 # 8-byte Spill
	movq	%rcx, 72(%rsp)                  # 8-byte Spill
	jge	.LBB1_37
# %bb.19:                               #   in Loop: Header=BB1_18 Depth=3
	xorl	%eax, %eax
	leaq	320(%rsp), %rbx
	movq	%r11, %r13
	jmp	.LBB1_20
	.p2align	4
.LBB1_37:                               #   in Loop: Header=BB1_18 Depth=3
	addq	%rdi, %r14
	movq	%r12, %rsi
	xorl	%r15d, %r15d
	leaq	320(%rsp), %rbx
	movq	%r11, %r13
	movq	296(%rsp), %rbp                 # 8-byte Reload
	.p2align	4
.LBB1_38:                               #   Parent Loop BB1_14 Depth=1
                                        #     Parent Loop BB1_16 Depth=2
                                        #       Parent Loop BB1_18 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	movq	%r14, 320(%rsp)
	movq	176(%rsp), %rax                 # 8-byte Reload
	addq	%r14, %rax
	movq	%rax, 328(%rsp)
	movq	504(%rsp), %rax                 # 8-byte Reload
	addq	%r14, %rax
	movq	%rax, 336(%rsp)
	movq	528(%rsp), %rax                 # 8-byte Reload
	addq	%r14, %rax
	movq	%rax, 344(%rsp)
	movl	84(%rsp), %ecx                  # 4-byte Reload
	leaq	368(%rsp), %rdx
	leaq	352(%rsp), %r8
	movq	%rbx, %r9
	callq	"??$q1t_dot@$01$03@@YAXHPEIAMPEBQEBUblock_q1_0@@PEBQEBUblock_q8_0@@@Z"
	movq	96(%rsp), %rdx                  # 8-byte Reload
	vmovss	368(%rsp), %xmm0                # xmm0 = mem[0],zero,zero,zero
	movq	72(%rsp), %rax                  # 8-byte Reload
	vmovss	%xmm0, (%rsi,%rax)
	vmovss	372(%rsp), %xmm0                # xmm0 = mem[0],zero,zero,zero
	movq	144(%rsp), %rax                 # 8-byte Reload
	vmovss	%xmm0, (%rsi,%rax)
	vmovss	376(%rsp), %xmm0                # xmm0 = mem[0],zero,zero,zero
	movq	312(%rsp), %rax                 # 8-byte Reload
	vmovss	%xmm0, (%rsi,%rax)
	vmovss	380(%rsp), %xmm0                # xmm0 = mem[0],zero,zero,zero
	movq	304(%rsp), %rax                 # 8-byte Reload
	vmovss	%xmm0, (%rsi,%rax)
	vmovss	384(%rsp), %xmm0                # xmm0 = mem[0],zero,zero,zero
	vmovss	%xmm0, (%rsi,%rdx)
	vmovss	388(%rsp), %xmm0                # xmm0 = mem[0],zero,zero,zero
	movq	88(%rsp), %rax                  # 8-byte Reload
	vmovss	%xmm0, (%rsi,%rax)
	vmovss	392(%rsp), %xmm0                # xmm0 = mem[0],zero,zero,zero
	vmovss	%xmm0, (%rsi,%r13)
	vmovss	396(%rsp), %xmm0                # xmm0 = mem[0],zero,zero,zero
	vmovss	%xmm0, (%rsi,%rbp)
	leaq	4(%r15), %rax
	addq	520(%rsp), %r14                 # 8-byte Folded Reload
	addq	512(%rsp), %rsi                 # 8-byte Folded Reload
	addq	$8, %r15
	cmpq	128(%rsp), %r15                 # 8-byte Folded Reload
	movq	%rax, %r15
	jle	.LBB1_38
.LBB1_20:                               #   in Loop: Header=BB1_18 Depth=3
	movq	%r13, 496(%rsp)                 # 8-byte Spill
	movq	128(%rsp), %r14                 # 8-byte Reload
	subq	%rax, %r14
	movl	84(%rsp), %esi                  # 4-byte Reload
	movq	280(%rsp), %r13                 # 8-byte Reload
	movq	176(%rsp), %r15                 # 8-byte Reload
	movq	480(%rsp), %rbp                 # 8-byte Reload
	jle	.LBB1_23
# %bb.21:                               #   in Loop: Header=BB1_18 Depth=3
	movq	%r13, %rcx
	imulq	%rax, %rcx
	addq	%rcx, %r12
	imulq	%r15, %rax
	addq	288(%rsp), %rax                 # 8-byte Folded Reload
	addq	%rax, %rdi
	.p2align	4
.LBB1_22:                               #   Parent Loop BB1_14 Depth=1
                                        #     Parent Loop BB1_16 Depth=2
                                        #       Parent Loop BB1_18 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	movq	352(%rsp), %r9
	movq	%rdi, 40(%rsp)
	movl	$1, 56(%rsp)
	movq	$0, 48(%rsp)
	movq	$0, 32(%rsp)
	movl	%esi, %ecx
	movq	%rbx, %rdx
	xorl	%r8d, %r8d
	callq	*%rbp
	vmovss	320(%rsp), %xmm0                # xmm0 = mem[0],zero,zero,zero
	movq	72(%rsp), %rax                  # 8-byte Reload
	vmovss	%xmm0, (%r12,%rax)
	movq	360(%rsp), %r9
	movq	%rdi, 40(%rsp)
	movl	$1, 56(%rsp)
	movq	$0, 48(%rsp)
	movq	$0, 32(%rsp)
	movl	%esi, %ecx
	movq	%rbx, %rdx
	xorl	%r8d, %r8d
	callq	*%rbp
	movq	96(%rsp), %rdx                  # 8-byte Reload
	vmovss	320(%rsp), %xmm0                # xmm0 = mem[0],zero,zero,zero
	vmovss	%xmm0, (%r12,%rdx)
	addq	%r13, %r12
	addq	%r15, %rdi
	decq	%r14
	jne	.LBB1_22
	jmp	.LBB1_23
.LBB1_26:
	movq	64(%rsp), %rax                  # 8-byte Reload
	addq	%rax, %rax
	cmpq	104(%rsp), %rax                 # 8-byte Folded Reload
	jge	.LBB1_36
# %bb.27:
	movq	%rax, %rsi
	movl	$41, %ecx
	callq	ggml_get_type_traits_cpu
	addq	152(%rsp), %rsi                 # 8-byte Folded Reload
	movq	%rsi, 64(%rsp)                  # 8-byte Spill
	cmpq	104(%rsp), %rsi                 # 8-byte Folded Reload
	jge	.LBB1_36
# %bb.28:
	movq	8(%rax), %r12
	movq	112(%rsp), %rax                 # 8-byte Reload
	movq	%rax, %rcx
	imulq	64(%rsp), %rcx                  # 8-byte Folded Reload
	movq	%rcx, 88(%rsp)                  # 8-byte Spill
	imulq	184(%rsp), %rax                 # 8-byte Folded Reload
	movq	%rax, 112(%rsp)                 # 8-byte Spill
	leaq	320(%rsp), %rsi
	movl	84(%rsp), %r15d                 # 4-byte Reload
	movq	176(%rsp), %r13                 # 8-byte Reload
	jmp	.LBB1_29
	.p2align	4
.LBB1_35:                               #   in Loop: Header=BB1_29 Depth=1
	movq	64(%rsp), %rax                  # 8-byte Reload
	addq	184(%rsp), %rax                 # 8-byte Folded Reload
	movq	88(%rsp), %rcx                  # 8-byte Reload
	addq	112(%rsp), %rcx                 # 8-byte Folded Reload
	movq	%rcx, 88(%rsp)                  # 8-byte Spill
	movq	%rax, 64(%rsp)                  # 8-byte Spill
	cmpq	104(%rsp), %rax                 # 8-byte Folded Reload
	jge	.LBB1_36
.LBB1_29:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_31 Depth 2
                                        #       Child Loop BB1_33 Depth 3
	cmpq	$0, 120(%rsp)                   # 8-byte Folded Reload
	jle	.LBB1_35
# %bb.30:                               #   in Loop: Header=BB1_29 Depth=1
	movq	64(%rsp), %r14                  # 8-byte Reload
	imulq	160(%rsp), %r14                 # 8-byte Folded Reload
	movq	192(%rsp), %rax                 # 8-byte Reload
	addq	248(%rax), %r14
	movq	$0, 72(%rsp)                    # 8-byte Folded Spill
	movq	88(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 96(%rsp)                  # 8-byte Spill
	xorl	%ecx, %ecx
	jmp	.LBB1_31
	.p2align	4
.LBB1_34:                               #   in Loop: Header=BB1_31 Depth=2
	movq	144(%rsp), %rcx                 # 8-byte Reload
	incq	%rcx
	movq	96(%rsp), %rax                  # 8-byte Reload
	addq	272(%rsp), %rax                 # 8-byte Folded Reload
	movq	%rax, 96(%rsp)                  # 8-byte Spill
	movq	72(%rsp), %rax                  # 8-byte Reload
	addq	264(%rsp), %rax                 # 8-byte Folded Reload
	movq	%rax, 72(%rsp)                  # 8-byte Spill
	cmpq	120(%rsp), %rcx                 # 8-byte Folded Reload
	je	.LBB1_35
.LBB1_31:                               #   Parent Loop BB1_29 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB1_33 Depth 3
	movq	%rcx, 144(%rsp)                 # 8-byte Spill
	cmpq	$0, 128(%rsp)                   # 8-byte Folded Reload
	jle	.LBB1_34
# %bb.32:                               #   in Loop: Header=BB1_31 Depth=2
	movq	168(%rsp), %rax                 # 8-byte Reload
	movq	248(%rax), %rbp
	addq	96(%rsp), %rbp                  # 8-byte Folded Reload
	movq	136(%rsp), %rax                 # 8-byte Reload
	movq	16(%rax), %rdi
	addq	72(%rsp), %rdi                  # 8-byte Folded Reload
	movq	128(%rsp), %rbx                 # 8-byte Reload
	.p2align	4
.LBB1_33:                               #   Parent Loop BB1_29 Depth=1
                                        #     Parent Loop BB1_31 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movq	%rdi, 40(%rsp)
	movl	$1, 56(%rsp)
	movq	$0, 48(%rsp)
	movq	$0, 32(%rsp)
	movl	%r15d, %ecx
	movq	%rsi, %rdx
	xorl	%r8d, %r8d
	movq	%r14, %r9
	callq	*%r12
	movq	280(%rsp), %rcx                 # 8-byte Reload
	vmovss	320(%rsp), %xmm0                # xmm0 = mem[0],zero,zero,zero
	vmovss	%xmm0, (%rbp)
	addq	%rcx, %rbp
	addq	%r13, %rdi
	decq	%rbx
	jne	.LBB1_33
	jmp	.LBB1_34
.LBB1_36:
	addq	$536, %rsp                      # imm = 0x218
	popq	%rbx
	popq	%rbp
	popq	%rdi
	popq	%rsi
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$q1t_dot@$01$03@@YAXHPEIAMPEBQEBUblock_q1_0@@PEBQEBUblock_q8_0@@@Z";
	.scl	2;
	.type	32;
	.endef
	.globl	__ymm@0303030303030303020202020202020201010101010101010000000000000000 # -- Begin function ??$q1t_dot@$01$03@@YAXHPEIAMPEBQEBUblock_q1_0@@PEBQEBUblock_q8_0@@@Z
	.section	.rdata,"dr",discard,__ymm@0303030303030303020202020202020201010101010101010000000000000000
	.p2align	5, 0x0
__ymm@0303030303030303020202020202020201010101010101010000000000000000:
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	1                               # 0x1
	.byte	1                               # 0x1
	.byte	1                               # 0x1
	.byte	1                               # 0x1
	.byte	1                               # 0x1
	.byte	1                               # 0x1
	.byte	1                               # 0x1
	.byte	1                               # 0x1
	.byte	2                               # 0x2
	.byte	2                               # 0x2
	.byte	2                               # 0x2
	.byte	2                               # 0x2
	.byte	2                               # 0x2
	.byte	2                               # 0x2
	.byte	2                               # 0x2
	.byte	2                               # 0x2
	.byte	3                               # 0x3
	.byte	3                               # 0x3
	.byte	3                               # 0x3
	.byte	3                               # 0x3
	.byte	3                               # 0x3
	.byte	3                               # 0x3
	.byte	3                               # 0x3
	.byte	3                               # 0x3
	.globl	__real@8040201008040201
	.section	.rdata,"dr",discard,__real@8040201008040201
	.p2align	3, 0x0
__real@8040201008040201:
	.byte	1                               # 0x1
	.byte	2                               # 0x2
	.byte	4                               # 0x4
	.byte	8                               # 0x8
	.byte	16                              # 0x10
	.byte	32                              # 0x20
	.byte	64                              # 0x40
	.byte	128                             # 0x80
	.globl	__ymm@0101010101010101010101010101010101010101010101010101010101010101
	.section	.rdata,"dr",discard,__ymm@0101010101010101010101010101010101010101010101010101010101010101
	.p2align	5, 0x0
__ymm@0101010101010101010101010101010101010101010101010101010101010101:
	.zero	32,1
	.globl	__ymm@0001000100010001000100010001000100010001000100010001000100010001
	.section	.rdata,"dr",discard,__ymm@0001000100010001000100010001000100010001000100010001000100010001
	.p2align	5, 0x0
__ymm@0001000100010001000100010001000100010001000100010001000100010001:
	.short	1                               # 0x1
	.short	1                               # 0x1
	.short	1                               # 0x1
	.short	1                               # 0x1
	.short	1                               # 0x1
	.short	1                               # 0x1
	.short	1                               # 0x1
	.short	1                               # 0x1
	.short	1                               # 0x1
	.short	1                               # 0x1
	.short	1                               # 0x1
	.short	1                               # 0x1
	.short	1                               # 0x1
	.short	1                               # 0x1
	.short	1                               # 0x1
	.short	1                               # 0x1
	.globl	__real@0000000400000000
	.section	.rdata,"dr",discard,__real@0000000400000000
	.p2align	3, 0x0
__real@0000000400000000:
	.long	0                               # 0x0
	.long	4                               # 0x4
	.section	.rdata,"dr"
	.p2align	1, 0x0
.LCPI2_5:
	.byte	1                               # 0x1
	.zero	1
.LCPI2_6:
	.short	1                               # 0x1
	.section	.text,"xr",discard,"??$q1t_dot@$01$03@@YAXHPEIAMPEBQEBUblock_q1_0@@PEBQEBUblock_q8_0@@@Z"
	.globl	"??$q1t_dot@$01$03@@YAXHPEIAMPEBQEBUblock_q1_0@@PEBQEBUblock_q8_0@@@Z"
	.p2align	4
"??$q1t_dot@$01$03@@YAXHPEIAMPEBQEBUblock_q1_0@@PEBQEBUblock_q8_0@@@Z": # @"??$q1t_dot@$01$03@@YAXHPEIAMPEBQEBUblock_q1_0@@PEBQEBUblock_q8_0@@@Z"
.seh_proc "??$q1t_dot@$01$03@@YAXHPEIAMPEBQEBUblock_q1_0@@PEBQEBUblock_q8_0@@@Z"
# %bb.0:
	pushq	%r15
	.seh_pushreg %r15
	pushq	%r14
	.seh_pushreg %r14
	pushq	%r13
	.seh_pushreg %r13
	pushq	%r12
	.seh_pushreg %r12
	pushq	%rsi
	.seh_pushreg %rsi
	pushq	%rdi
	.seh_pushreg %rdi
	pushq	%rbp
	.seh_pushreg %rbp
	pushq	%rbx
	.seh_pushreg %rbx
	subq	$312, %rsp                      # imm = 0x138
	.seh_stackalloc 312
	vmovaps	%xmm15, 288(%rsp)               # 16-byte Spill
	.seh_savexmm %xmm15, 288
	vmovdqa	%xmm14, 272(%rsp)               # 16-byte Spill
	.seh_savexmm %xmm14, 272
	vmovaps	%xmm13, 256(%rsp)               # 16-byte Spill
	.seh_savexmm %xmm13, 256
	vmovdqa	%xmm12, 240(%rsp)               # 16-byte Spill
	.seh_savexmm %xmm12, 240
	vmovaps	%xmm11, 224(%rsp)               # 16-byte Spill
	.seh_savexmm %xmm11, 224
	vmovaps	%xmm10, 208(%rsp)               # 16-byte Spill
	.seh_savexmm %xmm10, 208
	vmovdqa	%xmm9, 192(%rsp)                # 16-byte Spill
	.seh_savexmm %xmm9, 192
	vmovdqa	%xmm8, 176(%rsp)                # 16-byte Spill
	.seh_savexmm %xmm8, 176
	vmovaps	%xmm7, 160(%rsp)                # 16-byte Spill
	.seh_savexmm %xmm7, 160
	vmovaps	%xmm6, 144(%rsp)                # 16-byte Spill
	.seh_savexmm %xmm6, 144
	.seh_endprologue
                                        # kill: def $ecx killed $ecx def $rcx
	vxorps	%xmm10, %xmm10, %xmm10
	vpxor	%xmm0, %xmm0, %xmm0
	vmovdqu	%ymm0, 16(%rsp)                 # 32-byte Spill
	vxorps	%xmm15, %xmm15, %xmm15
	vxorps	%xmm1, %xmm1, %xmm1
	vxorps	%xmm5, %xmm5, %xmm5
	vxorps	%xmm4, %xmm4, %xmm4
	vxorps	%xmm13, %xmm13, %xmm13
	vxorps	%xmm7, %xmm7, %xmm7
	cmpl	$128, %ecx
	jl	.LBB2_5
# %bb.1:
	shrl	$7, %ecx
	movq	(%r8), %rax
	movq	8(%r8), %r8
	movq	(%r9), %r10
	movq	8(%r9), %r11
	movq	16(%r9), %rsi
	movq	24(%r9), %r9
	leaq	2(%rax), %rdi
	leaq	2(%r8), %rbx
	addq	$2, %r10
	addq	$2, %r11
	addq	$2, %rsi
	addq	$2, %r9
	xorl	%r14d, %r14d
	leaq	ggml_table_f32_f16(%rip), %r15
	vmovdqa	__ymm@0303030303030303020202020202020201010101010101010000000000000000(%rip), %ymm9 # ymm9 = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3]
	vpbroadcastq	__real@8040201008040201(%rip), %ymm14 # ymm14 = [1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128]
	vpbroadcastb	.LCPI2_5(%rip), %ymm12  # ymm12 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
	vmovups	%ymm1, 16(%rsp)                 # 32-byte Spill
	vmovdqu	%ymm14, 112(%rsp)               # 32-byte Spill
	vpbroadcastw	.LCPI2_6(%rip), %ymm8   # ymm8 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
	.p2align	4
.LBB2_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB2_3 Depth 2
	leaq	(%r14,%r14), %r12
	leaq	(%r12,%r12,8), %r12
	movzwl	(%rax,%r12), %r13d
	vmovss	(%r15,%r13,4), %xmm2            # xmm2 = mem[0],zero,zero,zero
	vmovss	%xmm2, 12(%rsp)                 # 4-byte Spill
	movzwl	(%r8,%r12), %r12d
	vmovss	(%r15,%r12,4), %xmm2            # xmm2 = mem[0],zero,zero,zero
	vmovss	%xmm2, 8(%rsp)                  # 4-byte Spill
	xorl	%r12d, %r12d
	xorl	%r13d, %r13d
	vmovups	16(%rsp), %ymm11                # 32-byte Reload
	vmovaps	%ymm15, %ymm3
	vmovaps	%ymm1, %ymm6
	.p2align	4
.LBB2_3:                                #   Parent Loop BB2_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vmovups	%ymm10, 48(%rsp)                # 32-byte Spill
	vmovups	%ymm7, 80(%rsp)                 # 32-byte Spill
	vpbroadcastd	(%rdi,%r13), %ymm7
	vpshufb	%ymm9, %ymm7, %ymm7
	vpand	%ymm7, %ymm14, %ymm7
	vpcmpeqb	%ymm0, %ymm7, %ymm7
	vpbroadcastd	(%rbx,%r13), %ymm15
	vpshufb	%ymm9, %ymm15, %ymm15
	vpand	%ymm14, %ymm15, %ymm15
	vpcmpeqb	%ymm0, %ymm15, %ymm9
	vmovdqu	(%r10,%r12), %ymm10
	movzwl	-2(%r10,%r12), %ebp
	vmovdqa	%ymm12, %ymm1
	vmovss	(%r15,%rbp,4), %xmm12           # xmm12 = mem[0],zero,zero,zero
	vpor	%ymm1, %ymm7, %ymm15
	vpsignb	%ymm15, %ymm10, %ymm7
	vpmaddubsw	%ymm7, %ymm1, %ymm7
	vpmaddwd	%ymm7, %ymm8, %ymm7
	vcvtdq2ps	%ymm7, %ymm7
	vmovss	12(%rsp), %xmm14                # 4-byte Reload
                                        # xmm14 = mem[0],zero,zero,zero
	vmulss	%xmm14, %xmm12, %xmm0
	vbroadcastss	%xmm0, %ymm0
	vfmadd231ps	%ymm0, %ymm7, %ymm6     # ymm6 = (ymm7 * ymm0) + ymm6
	vpor	%ymm1, %ymm9, %ymm7
	vpsignb	%ymm7, %ymm10, %ymm0
	vpmaddubsw	%ymm0, %ymm1, %ymm0
	vpmaddwd	%ymm0, %ymm8, %ymm0
	vcvtdq2ps	%ymm0, %ymm0
	vmovss	8(%rsp), %xmm2                  # 4-byte Reload
                                        # xmm2 = mem[0],zero,zero,zero
	vmulss	%xmm2, %xmm12, %xmm9
	vbroadcastss	%xmm9, %ymm9
	vfmadd231ps	%ymm9, %ymm0, %ymm5     # ymm5 = (ymm0 * ymm9) + ymm5
	vmovdqu	(%r11,%r12), %ymm0
	movzwl	-2(%r11,%r12), %ebp
	vmovss	(%r15,%rbp,4), %xmm9            # xmm9 = mem[0],zero,zero,zero
	vpsignb	%ymm15, %ymm0, %ymm10
	vpmaddubsw	%ymm10, %ymm1, %ymm10
	vpmaddwd	%ymm8, %ymm10, %ymm10
	vcvtdq2ps	%ymm10, %ymm10
	vmulss	%xmm14, %xmm9, %xmm12
	vbroadcastss	%xmm12, %ymm12
	vfmadd231ps	%ymm12, %ymm10, %ymm3   # ymm3 = (ymm10 * ymm12) + ymm3
	vpsignb	%ymm7, %ymm0, %ymm0
	vpmaddubsw	%ymm0, %ymm1, %ymm0
	vpmaddwd	%ymm0, %ymm8, %ymm0
	vcvtdq2ps	%ymm0, %ymm0
	vmulss	%xmm2, %xmm9, %xmm9
	vbroadcastss	%xmm9, %ymm9
	vfmadd231ps	%ymm9, %ymm0, %ymm4     # ymm4 = (ymm0 * ymm9) + ymm4
	vmovdqu	(%rsi,%r12), %ymm0
	movzwl	-2(%rsi,%r12), %ebp
	vmovss	(%r15,%rbp,4), %xmm9            # xmm9 = mem[0],zero,zero,zero
	vpsignb	%ymm15, %ymm0, %ymm10
	vpmaddubsw	%ymm10, %ymm1, %ymm10
	vpmaddwd	%ymm8, %ymm10, %ymm10
	vcvtdq2ps	%ymm10, %ymm10
	vmulss	%xmm14, %xmm9, %xmm12
	vbroadcastss	%xmm12, %ymm12
	vfmadd231ps	%ymm12, %ymm10, %ymm11  # ymm11 = (ymm10 * ymm12) + ymm11
	vpsignb	%ymm7, %ymm0, %ymm0
	vpmaddubsw	%ymm0, %ymm1, %ymm0
	vpmaddwd	%ymm0, %ymm8, %ymm0
	vcvtdq2ps	%ymm0, %ymm0
	vmulss	%xmm2, %xmm9, %xmm9
	vbroadcastss	%xmm9, %ymm9
	vfmadd231ps	%ymm9, %ymm0, %ymm13    # ymm13 = (ymm0 * ymm9) + ymm13
	vmovdqu	(%r9,%r12), %ymm0
	movzwl	-2(%r9,%r12), %ebp
	vmovss	(%r15,%rbp,4), %xmm9            # xmm9 = mem[0],zero,zero,zero
	vpsignb	%ymm15, %ymm0, %ymm10
	vpmaddubsw	%ymm10, %ymm1, %ymm10
	vpmaddwd	%ymm8, %ymm10, %ymm10
	vcvtdq2ps	%ymm10, %ymm10
	vmulss	%xmm14, %xmm9, %xmm12
	vbroadcastss	%xmm12, %ymm12
	vmovups	48(%rsp), %ymm14                # 32-byte Reload
	vfmadd231ps	%ymm12, %ymm10, %ymm14  # ymm14 = (ymm10 * ymm12) + ymm14
	vmovups	%ymm14, 48(%rsp)                # 32-byte Spill
	vmovdqa	%ymm1, %ymm12
	vmovups	48(%rsp), %ymm10                # 32-byte Reload
	vmovdqu	112(%rsp), %ymm14               # 32-byte Reload
	vpsignb	%ymm7, %ymm0, %ymm0
	vpmaddubsw	%ymm0, %ymm1, %ymm0
	vpmaddwd	%ymm0, %ymm8, %ymm0
	vcvtdq2ps	%ymm0, %ymm0
	vmulss	%xmm2, %xmm9, %xmm7
	vmovdqa	__ymm@0303030303030303020202020202020201010101010101010000000000000000(%rip), %ymm9 # ymm9 = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3]
	vbroadcastss	%xmm7, %ymm7
	vmovups	80(%rsp), %ymm1                 # 32-byte Reload
	vfmadd231ps	%ymm7, %ymm0, %ymm1     # ymm1 = (ymm0 * ymm7) + ymm1
	vmovups	%ymm1, 80(%rsp)                 # 32-byte Spill
	vmovups	80(%rsp), %ymm7                 # 32-byte Reload
	vxorps	%xmm0, %xmm0, %xmm0
	addq	$4, %r13
	addq	$34, %r12
	cmpq	$136, %r12
	jne	.LBB2_3
# %bb.4:                                #   in Loop: Header=BB2_2 Depth=1
	vmovaps	%ymm6, %ymm1
	vmovaps	%ymm3, %ymm15
	vmovups	%ymm11, 16(%rsp)                # 32-byte Spill
	incq	%r14
	addq	$18, %rdi
	addq	$18, %rbx
	addq	$136, %r10
	addq	$136, %r11
	addq	$136, %rsi
	addq	$136, %r9
	cmpq	%rcx, %r14
	jne	.LBB2_2
.LBB2_5:
	vmovaps	%ymm7, %ymm0
	vextractf128	$1, %ymm13, %xmm7
	vaddps	%xmm7, %xmm13, %xmm7
	vextractf128	$1, %ymm0, %xmm8
	vaddps	%xmm0, %xmm8, %xmm6
	vinsertf128	$1, %xmm6, %ymm7, %ymm6
	vhaddps	%ymm6, %ymm6, %ymm6
	vextractf128	$1, %ymm1, %xmm7
	vhaddps	%ymm6, %ymm6, %ymm6
	vaddps	%xmm7, %xmm1, %xmm9
	vbroadcastsd	__real@0000000400000000(%rip), %ymm7 # ymm7 = [0,4,0,4,0,4,0,4]
	vpermps	%ymm6, %ymm7, %ymm6
	vextractf128	$1, %ymm5, %xmm8
	vaddps	%xmm5, %xmm8, %xmm0
	vextractf128	$1, %ymm4, %xmm8
	vaddps	%xmm4, %xmm8, %xmm3
	vinsertf128	$1, %xmm3, %ymm0, %ymm3
	vhaddps	%ymm3, %ymm3, %ymm3
	vhaddps	%ymm3, %ymm3, %ymm3
	vextractf128	$1, %ymm15, %xmm4
	vaddps	%xmm4, %xmm15, %xmm2
	vhaddps	%xmm2, %xmm9, %xmm2
	vpermps	%ymm3, %ymm7, %ymm3
	vhaddps	%xmm2, %xmm2, %xmm2
	vmovups	16(%rsp), %ymm1                 # 32-byte Reload
	vextractf128	$1, %ymm1, %xmm4
	vaddps	%xmm4, %xmm1, %xmm1
	vextractf128	$1, %ymm10, %xmm4
	vaddps	%xmm4, %xmm10, %xmm0
	vinsertf128	$1, %xmm0, %ymm1, %ymm0
	vhaddps	%ymm0, %ymm0, %ymm0
	vblendps	$192, %ymm6, %ymm3, %ymm1       # ymm1 = ymm3[0,1,2,3,4,5],ymm6[6,7]
	vhaddps	%ymm0, %ymm0, %ymm0
	vpermps	%ymm0, %ymm7, %ymm0
	vblendps	$3, %xmm2, %xmm0, %xmm0         # xmm0 = xmm2[0,1],xmm0[2,3]
	vblendps	$240, %ymm1, %ymm0, %ymm0       # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
	vmovups	%ymm0, (%rdx)
	vmovaps	144(%rsp), %xmm6                # 16-byte Reload
	vmovaps	160(%rsp), %xmm7                # 16-byte Reload
	vmovaps	176(%rsp), %xmm8                # 16-byte Reload
	vmovaps	192(%rsp), %xmm9                # 16-byte Reload
	vmovaps	208(%rsp), %xmm10               # 16-byte Reload
	vmovaps	224(%rsp), %xmm11               # 16-byte Reload
	vmovaps	240(%rsp), %xmm12               # 16-byte Reload
	vmovaps	256(%rsp), %xmm13               # 16-byte Reload
	vmovaps	272(%rsp), %xmm14               # 16-byte Reload
	vmovaps	288(%rsp), %xmm15               # 16-byte Reload
	addq	$312, %rsp                      # imm = 0x138
	popq	%rbx
	popq	%rbp
	popq	%rdi
	popq	%rsi
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	vzeroupper
	retq
	.seh_endproc
                                        # -- End function
	.section	.bss,"bw",discard,_Avx2WmemEnabledWeakValue
	.globl	_Avx2WmemEnabledWeakValue       # @_Avx2WmemEnabledWeakValue
	.p2align	2, 0x0
_Avx2WmemEnabledWeakValue:
	.long	0                               # 0x0

	.lcomm	"?disabled@?1??ggml_cpu_q1t_supports_mul_mat@@9@4_NB",1 # @"?disabled@?1??ggml_cpu_q1t_supports_mul_mat@@9@4_NB"
	.lcomm	"?$TSS0@?1??ggml_cpu_q1t_supports_mul_mat@@9@4HA",4,4 # @"?$TSS0@?1??ggml_cpu_q1t_supports_mul_mat@@9@4HA"
	.section	.rdata,"dr",discard,"??_C@_0BA@PNIHBCGC@GGML_NO_Q1_TILE?$AA@"
	.globl	"??_C@_0BA@PNIHBCGC@GGML_NO_Q1_TILE?$AA@" # @"??_C@_0BA@PNIHBCGC@GGML_NO_Q1_TILE?$AA@"
"??_C@_0BA@PNIHBCGC@GGML_NO_Q1_TILE?$AA@":
	.asciz	"GGML_NO_Q1_TILE"

	.section	.drectve,"yni"
	.ascii	" /FAILIFMISMATCH:\"_CRT_STDIO_ISO_WIDE_SPECIFIERS=0\""
	.ascii	" /FAILIFMISMATCH:\"_MSC_VER=1900\""
	.ascii	" /FAILIFMISMATCH:\"_ITERATOR_DEBUG_LEVEL=0\""
	.ascii	" /FAILIFMISMATCH:\"RuntimeLibrary=MT_StaticRelease\""
	.ascii	" /DEFAULTLIB:libcpmt.lib"
	.ascii	" /alternatename:_Avx2WmemEnabled=_Avx2WmemEnabledWeakValue"
	.ascii	" /FAILIFMISMATCH:\"annotate_string=0\""
	.ascii	" /FAILIFMISMATCH:\"annotate_vector=0\""
	.ascii	" /FAILIFMISMATCH:\"annotate_optional=0\""
	.addrsig
	.addrsig_sym __CxxFrameHandler3
	.addrsig_sym "?disabled@?1??ggml_cpu_q1t_supports_mul_mat@@9@4_NB"
	.addrsig_sym "?$TSS0@?1??ggml_cpu_q1t_supports_mul_mat@@9@4HA"
	.globl	_fltused
