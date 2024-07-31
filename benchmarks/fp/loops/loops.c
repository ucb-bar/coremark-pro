/*
(C) 2014 EEMBC(R).  All rights reserved.

All EEMBC Benchmark Software are products of EEMBC
and are provided under the terms of the EEMBC Benchmark License Agreements.
The EEMBC Benchmark Software are proprietary intellectual properties of EEMBC and its Members
and is protected under all applicable laws, including all applicable copyright laws.
If you received this EEMBC Benchmark Software without having
a currently effective EEMBC Benchmark License Agreement, you must discontinue use.
Please refer to LICENSE.md for the specific license agreement that pertains to this Benchmark Software.
*/

/*************************
** Loops Tests **
** Perform a variety of small loops tests, taken from scimark and livermore loops.
*************************/
#include "th_cfg.h"
#include "th_math.h" /* for sin, cos and pow */
#include "th_lib.h"
#include "th_rand.h" /* initialize a random data vector */
#include "loops.h"

static int num_vectors=12;
static int num_2d_matrixes=9;
static int num_3d_matrixes=3;
#if (BMDEBUG && DEBUG_ACCURATE_BITS)
static int debug_counter=0;
#endif

loops_params presets_loops[NUM_DATAS];
extern void init_preset_0();
extern void init_preset_1();
extern void init_preset_2();
extern void init_preset_3();
extern void init_preset_4();
extern void init_preset_5();
extern void init_preset_6();
extern void init_preset_7();

/* ======================================================================== */
/*         FALSE U N C TRUE I O N   P R O TRUE O TRUE Y P E S                            */
/* ======================================================================== */
/* file provides :
define - init base input params (segment [within 0..2 boundary], number of coefficients to calculate, number of integration steps)
init -  allocate working memory, assign a[0] and b[0] .
run - calculate N coefficients in the segment
fini - calculate avg of coeffients then dealloc working memory
verify - SNR average of coefficients vs golden reference, must run at least base iterations for valid output.
clean - deallocate output memory
*/
void *define_params_loops(unsigned int idx, char *name, char *dataset);
void *bmark_init_loops(void *);
void *t_run_test_loops(struct TCDef *,void *);
int bmark_clean_loops(void *);
int bmark_verify_loops(void *in_params);
void *bmark_fini_loops(void *in_params);

/* benchmark function declarations */

e_fp MonteCarlo_integrate(loops_params *p);
e_fp hydro_fragment(loops_params *p);
e_fp cholesky(loops_params *p);
e_fp inner_product(loops_params *p);
e_fp banded_linear(loops_params *p);
e_fp tri_diagonal(loops_params *p);
e_fp linear_recurrence(loops_params *p);
e_fp state_fragment(loops_params *p);
e_fp adi_integration(loops_params *p);
e_fp integrate_predictors(loops_params *p);
e_fp difference_predictors(loops_params *p);
e_fp first_sum(loops_params *p);
e_fp first_dif(loops_params *p);
e_fp pic_2d(loops_params *par);
e_fp pic_1d(loops_params *par);
e_fp casual(loops_params *par);
e_fp monte_carlo(loops_params *par);
e_fp implicit(loops_params *par);
e_fp hydro_2d(loops_params *par);
e_fp lin_recurrence(loops_params *par);
e_fp ordinate_transport(loops_params *par);
e_fp matmul(loops_params *par);
e_fp planckian(loops_params *par);
e_fp hydro_2d_implicit(loops_params *par);
e_fp firstmin(loops_params *par);

typedef e_fp (*benchfunc)(loops_params *par);

benchfunc all_tests[] = {
	MonteCarlo_integrate,
	hydro_fragment,
	cholesky,
	inner_product,
	banded_linear,
	tri_diagonal,
	linear_recurrence,
	state_fragment,
	adi_integration,
	integrate_predictors,
	difference_predictors,
	first_sum,
	first_dif,
	pic_2d,
	pic_1d,
	casual,
	monte_carlo,
	implicit,
	hydro_2d,
	lin_recurrence,
	ordinate_transport,
	matmul,
	planckian,
	hydro_2d_implicit,
	firstmin,
	NULL
};
#if VERBOSE
char *test_type_str[] = {
	"MONTE_CARLO_PI_IDX			",
	"HYDRO_IDX					",
	"CHOLESKY_IDX				",
	"INNER_PRODUCT_IDX			",
	"BANDED_LINEAR_IDX			",
	"TRI_DIAGONAL_IDX			",
	"LINEAR_RECURRENCE_IDX		",
	"STATE_FRAGMENT_IDX			",
	"ADI_INTEGRATION_IDX		",
	"INTEGRATE_PREDICTORS_IDX	",
	"DIFFERENCE_PREDICTORS_IDX	",
	"FIRST_SUM_IDX				",
	"FIRST_DIF_IDX				",
	"PIC_2D_IDX					",
	"PIC_1D_IDX					",
	"CASUAL_IDX					",
	"MONTE_CARLO_IDX			",
	"IMPLICIT_IDX				",
	"HYDRO_2D_IDX				",
	"LIN_RECURRENCE_IDX			",
	"ORDINATE_TRANSPORT_IDX		",
	"MATMUL_IDX					",
	"PLANCKIAN_IDX				",
	"HYDRO_2D_IMPLICIT_IDX		",
	"FIRSTMIN_IDX				",
	"LAST_TEST_IDX				"
};
#endif

uint32_t hex(float f) {
  return *((uint32_t*)(&f));
}

e_fp *reinit_vec_limited(loops_params *params, e_fp *p, int nvals) {
	int i=0;
	int si=random_u32(params->r);
	while (i<nvals) {
		p[i]=params->lrbank[si++ & 0xfff];
		i++;
	}
	return p;
}

#define MIN(a,b)    ((a) < (b) ? (a) : (b))

e_fp *reinit_vec(loops_params *params, e_fp *p, int nvals) {
	int i=0;
	int si=random_u32(params->r);

#if USE_RVV
        si = si & 0xfff;
        while (i<nvals) {
          size_t vl = __riscv_vsetvl_e32m8(MIN(nvals - i, 0x1000 - si));
          vfloat32m8_t v = __riscv_vle32_v_f32m8(&params->rbank[si], vl);
          __riscv_vse32_v_f32m8(&p[i], v, vl);
          i += vl;
          si = (si + vl) & 0xfff;
        }
#else
        while (i<nvals) {
		p[i]=params->rbank[si++ & 0xfff];
		i++;
	}
#endif
	return p;
}
e_fp *reinit_vec_sparse(loops_params *params, e_fp *p, int nvals, int step) {
	int i=0;
	int si=random_u32(params->r);
	while (i<nvals) {
		p[i]=params->rbank[si++ & 0xfff];
		i+=step;
	}
	return p;
}
e_fp *reinit_ordinate_cvec(loops_params *params, e_fp *p, int nvals) {
        int i=0;
		int si=random_u32(params->r);
        while (i<nvals) {
                p[i]= FPCONST(1e-16)+EE_EPSILON*params->rbank[si++ & 0xfff];
                i++;
        }
        return p;
}

void reinit_ivec(loops_params *params, e_u32 *p, int nvals, e_u32 mask) {
	int i=0;
	int si=random_u32(params->r);
	while (i<nvals) {
		p[i]=params->irbank[si++ & 0xfff] & mask;
		i++;
	}
}
void reinit_ivec_pos(loops_params *params, e_u32 *p, int nvals, e_u32 mask) {
	int i=0;
	int si=random_u32(params->r);
	while (i<nvals) {
		p[i]=(params->irbank[si++ & 0xfff] & mask) + 1;
		i++;
	}
}
void reinit_ivec_sparse(loops_params *params, e_u32 *p, int nvals, e_u32 mask, int step) {
	int i=0;
	int si=random_u32(params->r);
	while (i<nvals) {
		p[i]=params->irbank[si++ & 0xfff] & mask;
		i+=step;
	}
}
void reinit_vec_const(loops_params *params, e_fp *p, int nvals, e_fp val) {
	int i;
#if USE_RVV
        size_t vl =__riscv_vsetvl_e32m8(nvals);
        vfloat32m8_t r = __riscv_vfmv_v_f_f32m8(val, vl);
        for (i=0 ; i<nvals; ) {
          vl = __riscv_vsetvl_e32m8(nvals - i);
          __riscv_vse32_v_f32m8(&p[i], r, vl);
          i += vl;
        }
#else
	for (i=0 ; i<nvals; i++)
		p[i]=val;
#endif
}
void zero_vec( e_fp *p, int nvals)
{
	reinit_vec_const(NULL,p,nvals,FPCONST(0.0));
}

static e_u32 pow2(e_u32 x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
	x++;
    return (x>>1);
}
static e_fp get_array_feedback(e_fp *a, int maxidx) {
	int i;
	e_fp ret=0.0;
	if (maxidx==0) return 0.0;
#if USE_RVV
        size_t vl;
        float vret;
        __asm__ volatile("vsetvli %0, zero, e32, m8, ta, ma" : "=r"(vl));
        __asm__ volatile("vmv.v.x  v0, %0" : : "r"(0x5555555555555555));
        __asm__ volatile("vmv.s.x v16, %0" : : "r"(0));
        for (i=0; i<maxidx; ) {
          __asm__ volatile("vsetvli  %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(maxidx-i));
          __asm__ volatile("vle32.v  v8, 0(%0)" : : "r"(&a[i]));
          __asm__ volatile("vfneg.v  v8, v8, v0.t");
          __asm__ volatile("vfredosum.vs v16, v8, v16");
          i += vl;
        }
        __asm__ volatile("vfmv.f.s %0, v16" : "=f"(vret));
        ret = vret;

#else
	for (i=0; i<maxidx; i++) {
                ret+=(i&1)?a[i]:-a[i];
		//PT: 4/4/2016: w/g decision
		//if (i&1) ret+=a[i];
		//else ret-=a[i];
	}
#endif

#if (BMDEBUG && DEBUG_ACCURATE_BITS)
	th_printf("\nfeed %d:",debug_counter++);
	th_print_fp(ret);
#endif
	return ret/(e_fp)maxidx;
}
/* Benchmark */
void *define_params_loops(unsigned int idx, char *name, char *dataset) {
    loops_params *params;
	e_u32 data_index=idx, req_bits=0;
	unsigned int default_disable_mask=0;

	th_memset(presets_loops,0,sizeof(presets_loops));
	init_preset_0();
	init_preset_1();
	init_preset_2();
	init_preset_3();
	init_preset_4();
	init_preset_5();
	init_preset_6();
	init_preset_7();

	/* parameter setup */
	params=(loops_params *)th_calloc(1,sizeof(loops_params));
	if ( params == NULL )
          th_exit( THE_OUT_OF_MEMORY, "Cannot Allocate Memory %s:%d", __FILE__,__LINE__ );

	params->gen_ref=0;
	params->seed=73686179;
	params->ref_data=NULL;
	params->N=1024;
	params->Loop=100;
	params->reinit_step=1;
	default_disable_mask = ~(FIRST_SUM|FIRST_DIF|FIRSTMIN|TRI_DIAGONAL|LINEAR_RECURRENCE|IMPLICIT|HYDRO_2D|INNER_PRODUCT);
	params->tests=0xffffffff & default_disable_mask;
	params->limit_int_input=0;

	if (pgo_training_run!=0) {
		data_index=4;
	} else {
		th_parse_buf_flag_unsigned(dataset,"-i",&data_index);
	}
	/* preset datasets */
	if (data_index<NUM_DATAS) {
		th_memcpy(params,&(presets_loops[data_index]),sizeof(loops_params));
	}
	/* command line overrides */
	if (pgo_training_run==0) {
		th_parse_buf_flag_unsigned(dataset,"-s",&params->seed);
		th_parse_buf_flag_unsigned(dataset,"-n",&params->N);
		th_parse_buf_flag_unsigned(dataset,"-x",&params->tests);
		th_parse_buf_flag(dataset,"-l",&params->Loop);
		th_parse_buf_flag(dataset,"-t",&params->reinit_step);
		th_parse_buf_flag(dataset,"-g",&params->gen_ref);
		th_parse_buf_flag_unsigned(dataset,"-b",&req_bits);
		th_parse_buf_flag_unsigned(dataset,"-r",&params->rtype);
		th_parse_buf_flag_unsigned(dataset,"-INT",&params->limit_int_input);
	}
	/* generic inits */
	{
		int i;
		void *r=NULL;
		switch (params->rtype) {
			case 3:
				r=rand_intparts_init(params->seed,256	  ,1	,-1	,7	,0			,EE_LIMIT_DYNAMIC_RANGE(0x00ffffff));
				params->rbank=fromint_random_fp_vector(0x1000,r); /* random bank -1..1 */
				break;
			case 2:
				params->rbank=fromint_fp_01_vector_uniform(0x1000,params->seed); /* random bank 0..1 */
				break;
			case 1:
				r=rand_intparts_init(params->seed,256	  ,1	,0	,0	,0			,EE_LIMIT_DYNAMIC_RANGE(0x0000ffff));
				params->rbank=fromint_random_fp_vector(0x1000,r); /* random bank, random sign */
				break;
			default:
				r=rand_intparts_init(params->seed,256	  ,0	,0	,0	,0			,EE_LIMIT_DYNAMIC_RANGE(0x0000ffff));
				params->rbank=fromint_random_fp_vector(0x1000,r); /* random bank with reduced probability of denormals */
				break;
		}
		for (i=0; i<0x1000; i++) {
			if (i&1) params->rbank[i]-=FPCONST(0.99);
			else params->rbank[i]-=FPCONST(1.01);
		}
		params->lrbank=(e_fp *)th_malloc(sizeof(e_fp)*0x1000);
		for (i=0; i<0x1000; i++) {
			params->lrbank[i]=(e_fp)((e_s32)random_u32_inrange(r,0,4) - (e_s32)2);
		}
		params->irbank=random_u32_vector(0x1000,params->seed);
		if (r)
			rand_fini(r);
	}
	{	int i; /* dataset sizes for the algorithms */
		int m2=(int)th_sqrt((e_fp)(params->N*4));
		for (i=0; i<32; i++) {
		params->nsize[i]=params->N;
		params->nruns[i]=params->Loop;
		}
		if (params->N>100)
			params->nsize[HYDRO_IDX]-=99;
		params->nsize[LINEAR_RECURRENCE_IDX]=m2;
		params->nsize[CASUAL_IDX]=params->N/7;
		params->nsize[FIRST_DIF_IDX]=params->N-1;
		params->nsize[HYDRO_2D_IDX]=params->N/7;
		params->nsize[MATMUL_IDX]=params->N/25;
		params->nsize[HYDRO_2D_IMPLICIT_IDX]=params->N/6;
		params->nsize[INTEGRATE_PREDICTORS_IDX]=params->N/13;
		params->nsize[DIFFERENCE_PREDICTORS_IDX]=params->N/14;
		params->nsize[CASUAL_IDX]=params->N/7;
		params->nsize[MONTE_CARLO_IDX]=m2/4;
		params->nsize[PIC_1D_IDX]=pow2(params->N);
	}

	params->vsize=params->N;
	params->vsize+=params->Loop;
	params->ivsize=params->N;
	params->m2size=params->N*4+4;
	params->m2size+=params->Loop;
	params->m3size=params->N*8;
	params->baseN=params->N;
	if (req_bits>0)
		params->minbits=req_bits;
	if (params->minbits==0) {
#if (USE_FP32)
	params->minbits=MIN_ACC_BITS_FP32;
#elif (USE_FP64)
	params->minbits=MIN_ACC_BITS_FP64;
#else
	params->minbits=MIN_ACC_BITS_OTHER;
#endif
	}
	return params;
}

int bmark_clean_loops(void *in_params) {
	loops_params *params=(loops_params *)in_params;
	if (params->irbank)
		th_free(params->irbank);
	if (params->rbank)
		th_free(params->rbank);
	if (params->lrbank)
		th_free(params->lrbank);
	th_free(params);
	return 1;
}

void reset_all_data(loops_params *p) {
	int i;
	for (i=0; i<num_vectors; i++) {
		zero_vec(p->v[i],p->vsize);
#if USE_RVV
                vec_memset(p->iv[i],0,p->ivsize);
#else
		th_memset(p->iv[i],0,p->ivsize);
#endif
	}
	for (i=0; i<num_2d_matrixes; i++) {
		zero_vec(p->m2[i],p->m2size);
	}
	for (i=0; i<num_3d_matrixes; i++) {
		zero_vec(p->m3[i],p->m3size);
	}
}
void dump_sparse_data(loops_params *p) {
	int i,j;
	for (i=0; i<num_vectors; i++) {
		th_printf("* v[%d]:\n",i);
		for (j=0; j<p->vsize; j+=p->reinit_step)
			th_print_fp(p->v[i][j]);
		th_printf("* iv[%d]:\n",i);
		for (j=0; j<p->ivsize; j+=p->reinit_step)
			th_printf("%08x,",p->iv[i][j]);
	}
	for (i=0; i<num_2d_matrixes; i++) {
		th_printf("* m2[%d]:\n",i);
		for (j=0; j<p->m2size; j+=p->reinit_step)
			th_print_fp(p->m2[i][j]);
	}
	for (i=0; i<num_3d_matrixes; i++) {
		th_printf("* m3[%d]:\n",i);
		for (j=0; j<p->m3size; j+=p->reinit_step)
			th_print_fp(p->m3[i][j]);
	}
}

void *bmark_init_loops(void *in_params) {
	/* Create a params for this invocation */
	int i;
	loops_params *params=(loops_params *)in_params;
    loops_params *myparams;

	if (in_params==NULL)
		th_exit( THE_BAD_PTR, "Invalid pointer %s:%d", __FILE__,__LINE__ );
	myparams=(loops_params *)th_calloc(1,sizeof(loops_params));
	if ( myparams == NULL )
        th_exit( THE_OUT_OF_MEMORY, "Cannot Allocate Memory %s:%d", __FILE__,__LINE__ );
	th_memcpy(myparams,params,sizeof(loops_params));
	/* Allocate working buffers */
	/* livermor has sizes 1001,101,2048,300 in space1 vectors */
	myparams->v=(e_fp **)th_malloc(sizeof(e_fp *)*num_vectors);
	for (i=0; i<num_vectors; i++) {
		int spacer;
		if (i<4) spacer=10;
		else spacer=0;
		myparams->v[i]=(e_fp *)th_malloc(sizeof(e_fp)*(myparams->vsize+spacer));
		if ( myparams->v[i] == NULL )
			th_exit( THE_OUT_OF_MEMORY, "Cannot Allocate Memory %s:%d", __FILE__,__LINE__ );
	}
	myparams->iv=(e_u32 **)th_malloc(sizeof(e_u32 *)*num_vectors);
	for (i=0; i<num_vectors; i++) {
		myparams->iv[i]=(e_u32 *)th_malloc(sizeof(e_u32)*myparams->ivsize);
		if ( myparams->iv[i] == NULL )
			th_exit( THE_OUT_OF_MEMORY, "Cannot Allocate Memory %s:%d", __FILE__,__LINE__ );
	}
	myparams->v2=(e_fp *)th_calloc(1,sizeof(e_fp)*(LAST_TEST_IDX+1));
	if ( myparams->v2 == NULL )
		th_exit( THE_OUT_OF_MEMORY, "Cannot Allocate Memory %s:%d", __FILE__,__LINE__ );

	myparams->m2=(e_fp **)th_malloc(sizeof(e_fp *)*num_2d_matrixes);
	for (i=0; i<num_2d_matrixes; i++) {
		myparams->m2[i]=(e_fp *)th_malloc(sizeof(e_fp)*myparams->m2size);
		if ( myparams->m2[i] == NULL )
			th_exit( THE_OUT_OF_MEMORY, "Cannot Allocate Memory %s:%d", __FILE__,__LINE__ );
	}
	myparams->m3=(e_fp **)th_malloc(sizeof(e_fp *)*num_3d_matrixes);
	for (i=0; i<num_3d_matrixes; i++) {
		myparams->m3[i]=(e_fp *)th_malloc(sizeof(e_fp)*myparams->m3size);
		if ( myparams->m3[i] == NULL )
			th_exit( THE_OUT_OF_MEMORY, "Cannot Allocate Memory %s:%d", __FILE__,__LINE__ );
	}
	reset_all_data(myparams);

	return myparams;
}


void *bmark_fini_loops(void *in_params) {
    loops_params *myparams;
	int i;
	if (in_params==NULL)
		th_exit( THE_BAD_PTR, "Invalid pointer %s:%d", __FILE__,__LINE__ );
	myparams=(loops_params *)in_params;
	/* Cleanup working buffers */
	if (myparams->v) {
	for (i=0; i<num_vectors; i++)
		th_free(myparams->v[i]);
	th_free(myparams->v);
	}
	if (myparams->v2) {
		th_free(myparams->v2);
	}
	if (myparams->iv) {
	for (i=0; i<num_vectors; i++)
		th_free(myparams->iv[i]);
	th_free(myparams->iv);
	}
	if (myparams->m2) {
	for (i=0; i<num_2d_matrixes; i++)
		th_free(myparams->m2[i]);
	th_free(myparams->m2);
	}
	if (myparams->m3) {
	for (i=0; i<num_3d_matrixes; i++)
		th_free(myparams->m3[i]);
	th_free(myparams->m3);
	}

	th_free(myparams);

	return NULL;
}

void *t_run_test_loops(struct TCDef *tcdef,void *in_params) {
	e_fp val=FPCONST(0.0);
	loops_params *params=(loops_params *)in_params;
	e_u32 next_test=1;
	e_u32 test_num=0;
	e_u32 test;
	//test_type run_test;

	tcdef->CRC=0;
	params->r=rand_init(params->seed,256,-1e10,1e10);
	while (test_num<LAST_TEST_IDX) {
		e_fp retval;
		while ((next_test & params->tests) == 0) {
			next_test<<=1;
			test_num++;
			if (test_num>=LAST_TEST_IDX) break;
		}
		if (all_tests[test_num]==NULL)
			break;
#if VERBOSE
		th_printf("Running %s\n",test_type_str[test_num]);
#endif
		params->N=params->nsize[test_num];
		params->Loop=params->nruns[test_num];
		retval=all_tests[test_num](params);
		params->v2[test_num]=retval;
		val+=retval;
		next_test<<=1;
		test_num++;
	}
	rand_fini(params->r);
	params->v2[LAST_TEST_IDX]=val;
	params->val=val;

	/* If ref data available, do a quick check */
	if (!params->gen_ref && params->ref_data) {
		test=fp_iaccurate_bits(val,&params->ref_data[LAST_TEST_IDX]);
		if (test>=params->minbits)
			tcdef->CRC=0;
		else
			tcdef->CRC=1;
		tcdef->v4=test;
	}

	/* Setup values */
	tcdef->v1=params->baseN;
	tcdef->v2=params->Loop;
	tcdef->v3=params->seed;
	/* Result values */
	tcdef->dbl_data[0]=params->val;

	return params;
}

int bmark_verify_loops(void *in_params) {
	int ret=1;
	loops_params *params=(loops_params *)in_params;

	if (params->gen_ref) {
		int i;
		th_printf("/**** START DATASET ****/\n#include \"th_lib.h\"\n#include \"../loops.h\"\n");
		th_printf("static intparts ref_data_index[]={\n");
		for (i=0; i<LAST_TEST_IDX+1; i++) {
			th_printf("\t");
			th_print_fp(params->v2[i]);
			if (i<LAST_TEST_IDX)
				th_printf(",\n");
		}
		th_printf("\n}; /* ref_data */\n\n");
		th_printf("void init_preset_index() {\n");
		th_printf("presets_loops[index].seed=0x%x;\n",params->seed);
		th_printf("presets_loops[index].N=0x%x;\n",params->baseN);
		th_printf("presets_loops[index].tests=0x%08x;\n",params->tests);
		th_printf("presets_loops[index].Loop=0x%x;\n",params->Loop);
		th_printf("presets_loops[index].minbits=%d;\n",params->minbits);
		th_printf("presets_loops[index].rtype=%d;\n",params->rtype);
		th_printf("presets_loops[index].limit_int_input=%u;\n",params->limit_int_input);
		th_printf("presets_loops[index].reinit_step=0x%x;\n",params->reinit_step);
		th_printf("presets_loops[index].ref_data=ref_data_index;\n}\n");
		th_printf("/**** END DATASET ****/\n");
	} else {
		snr_result resa;
		if (params->ref_data==NULL) {
			th_printf("Cannot validate this dataset! Please generate reference data on reliable hardware and tools using -g!\n");
			return 0;
		}
		resa.bmin_ok=params->minbits;
		ee_ifpbits_buffer(params->v2,params->ref_data,LAST_TEST_IDX,&resa);
#if (BMDEBUG)
		dump_sparse_data(params);
#endif
		if (!resa.pass) {
			th_printf("loops ERROR: a:%d,%d\n",resa.bmin,resa.bavg);
			return 0;
		}
	}
	return ret;
}

e_fp MonteCarlo_integrate(loops_params *p)
{
	int under_curve = 0;
	int outer;
	int inner;
	int Num_samples=p->Loop;
	int iloop=p->N;
	int oloop=Num_samples;
	e_fp *px=p->v[0], *py=p->v[1];

	for (outer=0; outer<oloop; outer++) {
		reinit_vec(p,px,iloop);
		reinit_vec(p,py,iloop);
#if USE_RVV
                for (inner=0; inner<iloop; ) {
                  size_t vl = __riscv_vsetvl_e32m8(iloop-inner);
                  vfloat32m8_t xs = __riscv_vle32_v_f32m8(&px[inner], vl);
                  vfloat32m8_t ys = __riscv_vle32_v_f32m8(&py[inner], vl);
                  vfloat32m8_t x2 = __riscv_vfmul_vv_f32m8(xs, xs, vl);
                  vfloat32m8_t r = __riscv_vfmadd_vv_f32m8(ys, ys, x2, vl);
                  vbool4_t b = __riscv_vmfle_vf_f32m8_b4(r, FPCONST(1.0), vl);
                  under_curve += __riscv_vcpop_m_b4(b, vl);
                  inner += iloop;
                }
#else
		for (inner=0; inner<iloop; inner++)
		{
			e_fp x=px[inner];
			e_fp y=py[inner];

			if ( x*x + y*y <= FPCONST(1.0))
				 under_curve ++;
		}
#endif
	}

	return ((e_fp) under_curve / Num_samples) * FPCONST(4.0);
}

e_fp hydro_fragment(loops_params *p) {
	e_fp *x=p->v[0], *y=p->v[1], *z=p->v[2];
	int n=p->N, loop=p->Loop,k,l;
	e_fp ret=FPCONST(0.0);
	e_fp q=p->v[3][0];
	e_fp r=p->v[3][1];
	e_fp t=p->v[3][2];
    /*
     *******************************************************************
     *   Kernel 1 -- hydro fragment
     *******************************************************************
     *       DO 1 L = 1,Loop
     *       DO 1 k = 1,n
     *  1       X(k)= Q + Y(k)*(R*ZX(k+10) + T*ZX(k+11))
	 *
	 * 	SG: Original code had only the last iteration of the outer loop mattered.
	 *	modified to get feedback from x between iterations.
     */

    for ( l=1 ; l<=loop ; l++ ) {
		reinit_vec(p,z,n+11); /* force multiple iterations with new data for each iteration */
        #if USE_RVV
        // Naive implementation: adds to calculate address, loads k+10 and k+11 separately, no static optimization
        uint32_t vl;
        // k is number of remaining elements
        // Don't need k to generate exact addresses due to unit-stride
        for (k=0 ; k<n ; k+=vl) {
            __asm__ volatile ("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n-k));
            __asm__ volatile ("vle32.v v16, (%0)" : : "r"(z+k+10)); //z[k+10]
            __asm__ volatile ("vfmul.vf v8, v16, %0" : : : "f"(r)); //r*
            __asm__ volatile ("vle32.v v24, (%0)" : : "r"(z+k+11)); //z[k+11]
            __asm__ volatile ("vfmacc.vf v8, v24, %" : : : "f"(t)); //+t*
            __asm__ volatile ("vle32.v v0, (%0)" : : "r"(y+k));     //y[k]
            __asm__ volatile ("vfmul.vv v16, v8, v0");              //y*()
            __asm__ volatile ("vfadd.vf v24, v16, %0" : : : "f"(q)) //q+
            __asm__ volatile ("vse32.v v0, (%0)" : : "r"(x+k));     //x[k]=
        }
        #else
        for ( k=0 ; k<n ; k++ ) {
            x[k] = q + y[k]*( r*z[k+10] + t*z[k+11] );
        }
        #endif
		ret+=get_array_feedback(x,n); /* force the calculation */
    }

	return ret;
}

e_fp cholesky(loops_params *p) {
	e_fp *x=p->v[0], *v=p->v[1];
	int n=p->N, loop=p->Loop;
	int ipntp, ipnt,ii,l,k,i;
	e_fp ret;
	reinit_vec(p,x,p->vsize);
	reinit_vec(p,v,p->vsize);
    /*
     *******************************************************************
     *   Kernel 2 -- ICCG excerpt (Incomplete Cholesky Conjugate Gradient)
     *******************************************************************
     *    DO 200  L= 1,Loop
     *        II= n
     *     IPNTP= 0
     *222   IPNT= IPNTP
     *     IPNTP= IPNTP+II
     *        II= II/2
     *         i= IPNTP
     CDIR$ IVDEP
     *    DO 2 k= IPNT+2,IPNTP,2
     *         i= i+1
     *  2   X(i)= X(k) - V(k)*X(k-1) - V(k+1)*X(k+1)
     *        IF( II.GT.1) GO TO 222
     *200 CONTINUE
     */

    for ( l=1 ; l<=loop ; l++ ) {
        ii = n/2;
        ipntp = 0;
        do {
            ipnt = ipntp;
            ipntp += ii;
            ii /= 2;
            i = ipntp - 1;
            #if USE_RVV
            // Could shift ii, but calculating ipntp requires many additions
            // ipntp-ipntp varies in length, so iterating over inner loop wastes less
            e_fp x_out = x+i;
            uint32_t vl = 1;
            for (k=ipnt+1; k<ipntp; k= k+(vl<<1)) {
                x_out += vl;
                __asm__ volatile ("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(k));

                __asm__ volatile ("vle32.v v0, (%0)" : : "r"(v+k));    //v[k]
                __asm__ volatile ("vle32.v v8, (%0)" : : "r"(x+k-1));  //x[k-1]
                __asm__ volatile ("vle32.v v16, (%0)" : : "r"(x+k));   //x[k]
                __asm__ volatile ("vfnmsac.vv v16, v0, v8");           //x[k]-v[k]*x[k-1]

                // Doesn't use shift on repeated array accesses
                // Loads will most likely be cached
                __asm__ volatile ("vle32.v v24, (%0)" : : "r"(x+k+1)); //x[k+1]
                __asm__ volatile ("vle32.v v8, (%0)" : : "r"(v+k+1));  //v[k+1]
                __asm__ volatile ("vfnmsac.vv v16, v8, v24");          //-v[k+1]*x[k+1]
                __asm__ volatile ("vse32.v v16, (%0)" : : "r"(x_out)); //x[i]=
            }

            #else
            for ( k=ipnt+1 ; k<ipntp ; k=k+2 ) {
                i++;
                x[i] = x[k] - v[k  ]*x[k-1] - v[k+1]*x[k+1];
            }
            #endif
        } while ( ii>0 );
    }
	if (p->full_verify)
		ret=get_array_feedback(x,n);
	else
		ret=x[0];

	return ret;
}

e_fp inner_product(loops_params *p) {
	e_fp *x=p->v[0],  *z=p->v[1];
	int n=p->N, loop=p->Loop;
	int l,k;
    e_fp q = FPCONST(0.0);
	reinit_vec_limited(p,x,p->vsize);
	reinit_vec_limited(p,z,p->vsize);

    /*
     *******************************************************************
     *   Kernel 3 -- inner product
     *******************************************************************
     *    DO 3 L= 1,Loop
     *         Q= 0.0
     *    DO 3 k= 1,n
     *  3      Q= Q + Z(k)*X(k)
	 *
	 *	SG : original code only had to perform one loop. Moved q initialization out of loop.
     */
    for ( l=1 ; l<=loop ; l++ ) {
        #if USE_RVV
        // Directly track current addresses
        e_fp* z_step = z+l, x_step = x;
        uint32_t vl;
        __asm__ volatile ("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));
        __asm__ volatile ("vfmv.s.f v2, %0" : : "f"(q));
        // k is number of remaining elements
        // Don't need k to generate exact addresses due to unit-stride
        for (k=n; k>0; k-=vl) {
            __asm__ volatile ("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(k));
            __asm__ volatile ("vle32.v v16, (%0)" : : "r"(z_step));
            __asm__ volatile ("vle32.v v24, (%0)" : : "r"(x_step));
            __asm__ volatile ("vfmul.vv v8, v16, v24");
            // Operations must be in-order for exact replication, so must reduce in loop
            __asm__ volatile ("vfredosum.vs v2, v8, v2");
            z_step+=vl, x_step+=vl;
        }
        __asm__ ("vfmv.f.s %0, v2" : "=f" (q));

        #else
        for ( k=0 ; k<n ; k++ ) {
            q += z[k+l]*x[k];
        }
        #endif

        #if (BMDEBUG && DEBUG_ACCURATE_BITS)
	    th_printf("\niprod %d:",debug_counter++);
	    th_print_fp(q);
        #endif
    }
	return q;
}

e_fp banded_linear(loops_params *p) {
	e_fp *x=p->m2[0]; /* really a vector, but needs size N+1001, so use matrix memory instead */
	e_fp *y=p->v[1];
	int n=p->N, loop=p->Loop;
	int m,l,lw,j,k;
	e_fp temp,ret;
	reinit_vec(p,x,p->vsize);
	reinit_vec(p,y,p->vsize);

	if (p->N<1001) {
		th_exit(THE_FAILURE,"Bad size for banded linear equations, must be at least 1001");
	}
    /*
     *******************************************************************
     *   Kernel 4 -- banded linear equations
     *******************************************************************
     *            m= (1001-7)/2
     *    DO 444  L= 1,Loop
     *    DO 444  k= 7,1001,m
     *           lw= k-6
     *         temp= X(k-1)
     CDIR$ IVDEP
     *    DO   4  j= 5,n,5
     *       temp  = temp   - XZ(lw)*Y(j)
     *  4        lw= lw+1
     *       X(k-1)= Y(5)*temp
     *444 CONTINUE
     */

    m = ( 1001-7 )/2;
    for ( l=1 ; l<=loop ; l++ ) {
        for ( k=6 ; k<1001 ; k=k+m ) {
            #if USE_RVV
            // Innermost loop vectorized because outer only does a few iterations
            unsigned int vl;
            const unsigned int ystride = 5*sizeof(e_fp);
            unsigned int remaining = (n-4)/5;
            e_fp* xptr = x+(k-6);
            __asm__ volatile("vsetvli %0, %1, e32, m8, ta, mu" : "=r"(vl), "r"(1));
            __asm__ volatile("vfmv.s.f v0, %0" : "f"(x[k-1]));
            for (j=4; j<n; j+=5*vl) {
                __asm__ volatile("vsetvli %0, %1, e32, m8, ta, mu" : "=r"(vl), "r"(remaining));
                __asm__ volatile("vle32.v v8, %0": "r"(xptr)); //x[lw]
                __asm__ volatile("vlse32.v v16, (%0), %1" : "r"(y+j), "r"(ystride)); //y[j]
                __asm__ volatile("vfmul.vv v32, v8, v16"); //x[lw]*y[j]
                __asm__ volatile("vfrsub.vf v32, v32, %0" : "f"(0)); //-x[lw]*y[j]
                __asm__ volatile("vfredosum.v v0, v32, v0"); //temp += -x[lw]*y[j]
                xptr += vl;
                remaining -= vl;
            }
            __asm__ volatile("vfmv.f.s %0, v0" : "f"(x[k-1]));
            #else
            lw = k - 6;
            temp = x[k-1];
            for ( j=4 ; j<n ; j=j+5 ) {
                temp -= x[lw]*y[j];
                lw++;
            }
            #endif
            x[k-1] = y[4]*temp;
        }
    }
	if (p->full_verify)
		ret=get_array_feedback(x,n);
	else
		ret=x[0];

	return ret;
}

e_fp tri_diagonal(loops_params *p) {
	e_fp *x=p->v[0],  *y=p->v[1],  *z=p->v[2];
	int n=p->N, loop=p->Loop;
	int i,l;
	e_fp ret;
	reinit_vec(p,x,n);
	reinit_vec(p,y,n);
	reinit_vec(p,z,n);
    /*
     *******************************************************************
     *   Kernel 5 -- tri-diagonal elimination, below diagonal
     *******************************************************************
     *    DO 5 L = 1,Loop
     *    DO 5 i = 2,n
     *  5    X(i)= Z(i)*(Y(i) - X(i-1))
	 *
	 *	SG: Original code only had to execute one loop. Modified with carry over to first index.
     */

    for ( l=1 ; l<=loop ; l++ ) {
        for ( i=1 ; i<n ; i++ ) {
            x[i] = z[i]*( y[i] - x[i-1] );
			if (!th_isfinite(x[i])) x[i]=y[i]; /* modified to avoid potential inf */
        }
		x[0]=z[0]*( y[0] - x[n-1] );
    }
	if (p->full_verify)
		ret=get_array_feedback(x,n);
	else
		ret=x[0];

	return ret;
}

e_fp linear_recurrence(loops_params *p) {
	e_fp *w=p->v[0],  *b=p->m2[0];
	int n=p->N, loop=p->Loop;
	int i,l,k=0;
	e_fp ret;
	reinit_vec(p,b,n*n);
    /*
     *******************************************************************
     *   Kernel 6 -- general linear recurrence equations
     *******************************************************************
     *    DO  6  L= 1,Loop
     *    DO  6  i= 2,n
     *    DO  6  k= 1,i-1
     *        W(i)= W(i)  + B(i,k) * W(i-k)
     *  6 CONTINUE
     */
    for ( i=0 ; i<n ; i++ ) {
		w[i]=FPCONST(0.0100);
	}
    for ( l=1 ; l<=loop ; l++ ) {
        for ( i=1 ; i<n ; i++ ) {
            #if USE_RVV
            //FIXME: This code was ported incorrectly and is missing the += to add
            // W(i) to itself in the inner loop instead of repeatedly reassigning it.
            // Once the output checks are updated, vectorizing the inner loop will be more efficient.
            // If !isfinite check changed, also fix sum loop.
            w[i] = b[(i-1)*n+i] * w[0];
			if (!th_isfinite(w[i])) w[i]=b[(i-1)*n+i]; /* avoid potential inf */

            /*
            // Cannot vectorize over i because each step uses all previous w[i]
            // Instead, vectorize over k to perform sum of products
            e_fp* bstart = b+i, wstart = w+i-1;
            uint32_t vl;
            e_fp sum;
            const int bstride = n*sizeof(e_fp), wstride = -sizeof(e_fp); // Strides for k*n, -k (in bytes) as k increments
            for (k=0; k<i; k+=vl) {
                __asm__ volatile ("vsetvli %0, %1, e32, m8, ta, mu" : "=r"(vl) : "r"(k-i));
                // Multiply inputs from b and w
                __asm__ volatile("vlse32.v v0, (%0), %1" : : "r"(b+i) :"r"(bstride)); //b[k*n+i]
                __asm__ volatile("vlse32.v v16, (%0), %1" : : "r"(w) :"r"(wstride)); //w[(i-k)-1]
                __asm__ volatile("vfmul.vv v16, v16, v0");

                // Sum
                // Currently the result and not addend is overwritten if infinite.
                // Need to check every sum in case one is denormal.
                // If this were changed, would do a masked vmv and vfredosum instead of loop.
                uint32_t curr_b_elem = 0;
                for (int j = 0; j<vl; j++) {
                    __asm__ volatile("vfmv.f.s %0, v16" : "=f"(sum));
                    w[i] += sum;
                    if (!th_isfinite(w[i])) {
                        __asm__ volatile("vslidedown.vx v0, v0, %0") : : : "=r"(j-curr_b_elem);
                        __asm__ volatile("vfmv.f.s %0, v0" : "=f"(sum));
                        w[i] = sum;
                        curr_b_elem = j;
                    }
                    // Get next term for sum
                    __asm__ volatile("vslidedown.vi v16, v16, 1");
                }
                
                wstart -= vl;
                bstart += vl*k;
            }*/
            #else
            for ( k=0 ; k<i ; k++ ) {
                w[i] = b[k*n+i] * w[(i-k)-1];
				if (!th_isfinite(w[i])) w[i]=b[k*n+i]; /* avoid potential inf */
            }
            #endif
        }
		w[0]=b[k*n] * w[n-1];
    }
	if (p->full_verify)
		ret=get_array_feedback(w,n);
	else
		ret=w[0];

	return ret;
}

e_fp state_fragment(loops_params *p) {
	e_fp *u=p->v[0],  *y=p->v[1],  *z=p->v[2];
	e_fp *x=p->v[3];
	int n=p->N, loop=p->Loop;
	int k,l;
	e_fp r=FPCONST(0.3);
	e_fp t=FPCONST(0.25);
	e_fp ret=FPCONST(0.0);
	reinit_vec(p,y,n);
	reinit_vec(p,z,n);
    /*
     *******************************************************************
     *   Kernel 7 -- equation of state fragment
     *******************************************************************
     *    DO 7 L= 1,Loop
     *    DO 7 k= 1,n
     *      X(k)=     U(k  ) + R*( Z(k  ) + R*Y(k  )) +
     *   .        T*( U(k+3) + R*( U(k+2) + R*U(k+1)) +
     *   .        T*( U(k+6) + R*( U(k+5) + R*U(k+4))))
     *  7 CONTINUE
	 *
	 *	SG: Had to modify since outer loop only had to execute once if compiler is smart enough
     */
    for ( l=1 ; l<=loop ; l++ ) {
      reinit_vec(p,u,n+6);
      #if USE_RVV
      uint32_t vl;
      float u1, u2, u3;
      for (k=0; k<n; k+=vl) {
        __asm__ volatile ("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n-k));

        // u[k] + r*( z[k] + r*y[k] )
        __asm__ volatile ("vle32.v v0, (%0)" : : "r"(y+k)); //y[k]
        __asm__ volatile ("vle32.v v8, (%0)" : : "r"(z+k)); //z[k]
        __asm__ volatile ("vfmadd.vf v0, v8, %0" : : : "f"(r)); //A= z[k]+r*y[k]
        __asm__ volatile ("vle32.v v16, (%0)" : : "r"(u+k)); //u[k]
        __asm__ volatile ("vfmadd.vf v0, v16, %0" : : : "f"(r)); //B= u[k]+r*A

        // + t*( u[k+3] + r*(u[k+2] + r*u[k+1])
        // Load last scalar of u[k+1] vector and use u[k] for the rest
        // Alt implementation: Load u+vl to u+vl+5 as a vector and slide appropriate parts
        // but requires two slide operations for u+#
        u1 = u[k+vl];
        __asm__ volatile ("vfslide1down.vf v16, v16, (%0)" : : : "f"(u1)); //u[k+1]
        u2 = u[k+vl+1];
        __asm__ volatile ("vfslide1down.vf v24, v16, (%0)" : : : "f"(u2)); //u[k+2]
        __asm__ volatile ("vfmadd.vf v16, v24, %0" : : : "f"(r)); // C= u[k+2] + r*u[k+1]
        u3 = u[k+vl+2];
        __asm__ volatile ("vfslide1down.vf v8, v24, (%0)" : : : "f"(u3)); //u[k+3]
        __asm__ volatile ("vfmadd.vf v16, v8, %0" : : : "f"(r)); // D= u[k+3]+r*C
        __asm__ volatile ("vfmacc.vf v0, v16, %0" : : : "f"(t)); // B+= t*D

        
        // +t*( u[k+6] + r*( u[k+5] + r*u[k+4])))
        u1 = u[k+vl+3];
        __asm__ volatile ("vfslide1down.vf v8, v8, (%0)" : : : "f"(u1)); //u[k+4]
        u2 = u[k+vl+4];
        __asm__ volatile ("vfslide1down.vf v16, v8, (%0)" : : : "f"(u2)); //u[k+5]
        __asm__ volatile ("vfmadd.vf v8, v16, %0" : : : "f"(r)); // E= u[k+5] + r*u[k+4]
        u3 = u[k+vl+5];
        __asm__ volatile ("vfslide1down.vf v24, v16, (%0)" : : : "f"(u3)); //u[k+6]
        __asm__ volatile ("vfmacc.vf v24, v8, %0" : : : "f"(r)); // F= u[k+6] + r*E
        __asm__ volatile ("vfmacc.vf v0, v24, %0" : : : "f"(t)); // B+= t*F

        __asm__ volatile ("vse32.v v0, (%0)" : : "r"(x+k)); // x[k]= B
      }
      #else
      for ( k=0 ; k<n ; k++ ) {
	x[k] = u[k] + r*( z[k] + r*y[k] ) +
	  t*( u[k+3] + r*( u[k+2] + r*u[k+1] ) +
	      t*( u[k+6] + r*( u[k+5] + r*u[k+4] ) ) );
      }
      #endif
      ret+=get_array_feedback(x,n);
    }

	return ret;
}

e_fp adi_integration(loops_params *p) {
	e_fp *du1=p->v[0],  *du2=p->v[1],  *du3=p->v[2]; /* temporary outputs */
	e_fp *u1=p->m3[0],  *u2=p->m3[1],  *u3=p->m3[2];
	int n=p->N, loop=p->Loop;
	int nl2,kx,ky,l;
	e_fp ret=FPCONST(0.0);
	e_fp *v=reinit_vec(p,p->v[3],10); /* get a few radom values for the a[i] */
	e_fp a11=v[0];
	e_fp a12=v[1];
	e_fp a13=v[2];
	e_fp a21=v[3];
	e_fp a22=v[4];
	e_fp a23=v[5];
	e_fp a31=v[6];
	e_fp a32=v[7];
	e_fp a33=v[8];
	e_fp sig=v[9];;
    /*
     *******************************************************************
     *   Kernel 8 -- ADI integration
     *******************************************************************
     *    DO  8      L = 1,Loop
     *             nl1 = 1
     *             nl2 = 2
     *    DO  8     kx = 2,3
     CDIR$ IVDEP
     *    DO  8     ky = 2,n
     *          DU1(ky)=U1(kx,ky+1,nl1)  -  U1(kx,ky-1,nl1)
     *          DU2(ky)=U2(kx,ky+1,nl1)  -  U2(kx,ky-1,nl1)
     *          DU3(ky)=U3(kx,ky+1,nl1)  -  U3(kx,ky-1,nl1)
     *    U1(kx,ky,nl2)=U1(kx,ky,nl1) +A11*DU1(ky) +A12*DU2(ky) +A13*DU3(ky)
     *   .       + SIG*(U1(kx+1,ky,nl1) -2.*U1(kx,ky,nl1) +U1(kx-1,ky,nl1))
     *    U2(kx,ky,nl2)=U2(kx,ky,nl1) +A21*DU1(ky) +A22*DU2(ky) +A23*DU3(ky)
     *   .       + SIG*(U2(kx+1,ky,nl1) -2.*U2(kx,ky,nl1) +U2(kx-1,ky,nl1))
     *    U3(kx,ky,nl2)=U3(kx,ky,nl1) +A31*DU1(ky) +A32*DU2(ky) +A33*DU3(ky)
     *   .       + SIG*(U3(kx+1,ky,nl1) -2.*U3(kx,ky,nl1) +U3(kx-1,ky,nl1))
     *  8 CONTINUE
     */

    for ( l=1 ; l<=loop ; l++ ) {
      reinit_vec(p,u1,(n+1)*3);
      reinit_vec(p,u2,(n+1)*3);
      reinit_vec(p,u3,(n+1)*3);
      nl2 = 1;
#if USE_RVV
      for (ky = 1; ky < n; ) {
        size_t vl;
        int curb = (ky)*3;
        int ioutb = nl2*n*3+ky*3;
        __asm__ volatile("vsetvli %0, %0, e32, m1, ta, ma" : "=r"(vl) : "r"(n - ky));
        //         -2    -1     0   1     2    3    4    5
        // u1: ( ,  v6,  v7) ( v3,  v4,  v5) ( v0,  v1,  v2)
        // u2: ( , v14, v15) (v11, v12, v13) ( v8,  v9, v10)
        // u3: ( , v22, v23) (v19, v20, v21) (v16, v17, v18)
        __asm__ volatile("vlseg3e32.v     v0, (%0)"    : : "r"(&u1[curb+3]));
        __asm__ volatile("vfslide1up.vf   v3,  v0, %0" : : "f"(u1[curb+0]));
        __asm__ volatile("vfslide1up.vf   v4,  v1, %0" : : "f"(u1[curb+1]));
        __asm__ volatile("vfslide1up.vf   v5,  v2, %0" : : "f"(u1[curb+2]));
        __asm__ volatile("vfslide1up.vf   v6,  v4, %0" : : "f"(u1[curb-2]));
        __asm__ volatile("vfslide1up.vf   v7,  v5, %0" : : "f"(u1[curb-1]));

        __asm__ volatile("vlseg3e32.v     v8, (%0)"    : : "r"(&u2[curb+3]));
        __asm__ volatile("vfslide1up.vf  v11,  v8, %0" : : "f"(u2[curb+0]));
        __asm__ volatile("vfslide1up.vf  v12,  v9, %0" : : "f"(u2[curb+1]));
        __asm__ volatile("vfslide1up.vf  v13, v10, %0" : : "f"(u2[curb+2]));
        __asm__ volatile("vfslide1up.vf  v14, v12, %0" : : "f"(u2[curb-2]));
        __asm__ volatile("vfslide1up.vf  v15, v13, %0" : : "f"(u2[curb-1]));

        __asm__ volatile("vlseg3e32.v    v16, (%0)"    : : "r"(&u3[curb+3]));
        __asm__ volatile("vfslide1up.vf  v19, v16, %0" : : "f"(u3[curb+0]));
        __asm__ volatile("vfslide1up.vf  v20, v17, %0" : : "f"(u3[curb+1]));
        __asm__ volatile("vfslide1up.vf  v21, v18, %0" : : "f"(u3[curb+2]));
        __asm__ volatile("vfslide1up.vf  v22, v20, %0" : : "f"(u3[curb-2]));
        __asm__ volatile("vfslide1up.vf  v23, v21, %0" : : "f"(u3[curb-1]));

        __asm__ volatile("vfsub.vv       v24,  v1,  v6");                       // d11: v24
        __asm__ volatile("vfsub.vv       v25,  v9, v14");                       // d21: v25
        __asm__ volatile("vfsub.vv       v26, v17, v22");                       // d31: v26
        __asm__ volatile("vfsub.vv       v27,  v2,  v7");                       // d12: v27
        __asm__ volatile("vfsub.vv       v28, v10, v15");                       // d22: v28
        __asm__ volatile("vfsub.vv       v29, v18, v23");                       // d33: v29

        __asm__ volatile("vmv.v.v        v30,  v5");
        __asm__ volatile("vfmacc.vf      v30,  %0,  v4": : "f"(-FPCONST(2.0))); //                                          u1p[2]-FPCONST(2.0)*u1p[1]        ;
        __asm__ volatile("vfadd.vv        v3, v30,  v3");                       //                                                                    +u1p[0] ;
        __asm__ volatile("vmv.v.v        v30,  v4");
        __asm__ volatile("vfmacc.vf      v30,  %0, v24": : "f"(a11));           // u1p[1]+a11*du11                                                            ;
        __asm__ volatile("vfmacc.vf      v30,  %0, v25": : "f"(a12));           //                +a12*du21                                                   ;
        __asm__ volatile("vfmacc.vf      v30,  %0, v26": : "f"(a13));           //                         +a13*du31                                          ;
        __asm__ volatile("vfmacc.vf      v30,  %0,  v3": : "f"(sig));           //                                   + sig*(                                 );

        __asm__ volatile("vfmacc.vf       v0,  %0,  v5": : "f"(-FPCONST(2.0))); //                                          u1p[3]-FPCONST(2.0)*u1p[2]        ;
        __asm__ volatile("vfadd.vv        v0,  v0,  v4");                       //                                                                    +u1p[1] ;
        __asm__ volatile("vmv.v.v        v31,  v5");
        __asm__ volatile("vfmacc.vf      v31,  %0, v27": : "f"(a11));           // u1p[2]+a11*du12                                                            ;
        __asm__ volatile("vfmacc.vf      v31,  %0, v28": : "f"(a12));           //                +a12*du22                                                   ;
        __asm__ volatile("vfmacc.vf      v31,  %0, v29": : "f"(a13));           //                         +a13*du32                                          ;
        __asm__ volatile("vfmacc.vf      v31,  %0,  v0": : "f"(sig));           //                                   + sig*(                                 );

        __asm__ volatile("vssseg2e32.v   v30, (%0), %1": : "r"(&u1[ioutb+1]), "r"(3*sizeof(float)));

        __asm__ volatile("vmv.v.v        v30, v13");
        __asm__ volatile("vfmacc.vf      v30,  %0, v12": : "f"(-FPCONST(2.0))); //                                          u2p[2]-FPCONST(2.0)*u2p[1]        ;
        __asm__ volatile("vfadd.vv       v11, v30, v11");                       //                                                                    +u2p[0] ;
        __asm__ volatile("vmv.v.v        v30, v12");
        __asm__ volatile("vfmacc.vf      v30,  %0, v24": : "f"(a21));           // u2p[1]+a21*du11                                                            ;
        __asm__ volatile("vfmacc.vf      v30,  %0, v25": : "f"(a22));           //                +a22*du21                                                   ;
        __asm__ volatile("vfmacc.vf      v30,  %0, v26": : "f"(a23));           //                         +a23*du31                                          ;
        __asm__ volatile("vfmacc.vf      v30,  %0, v11": : "f"(sig));           //                                   + sig*(                                 );

        __asm__ volatile("vfmacc.vf       v8,  %0, v13": : "f"(-FPCONST(2.0))); //                                          u2p[3]-FPCONST(2.0)*u2p[2]        ;
        __asm__ volatile("vfadd.vv        v8,  v8, v12");                       //                                                                    +u2p[1] ;
        __asm__ volatile("vmv.v.v        v31, v13");
        __asm__ volatile("vfmacc.vf      v31,  %0, v27": : "f"(a21));           // u2p[2]+a21*du12                                                            ;
        __asm__ volatile("vfmacc.vf      v31,  %0, v28": : "f"(a22));           //                +a22*du22                                                   ;
        __asm__ volatile("vfmacc.vf      v31,  %0, v29": : "f"(a23));           //                         +a23*du32                                          ;
        __asm__ volatile("vfmacc.vf      v31,  %0,  v8": : "f"(sig));           //                                   + sig*(                                 );

        __asm__ volatile("vssseg2e32.v   v30, (%0), %1": : "r"(&u2[ioutb+1]), "r"(3*sizeof(float)));

        __asm__ volatile("vmv.v.v        v30, v21");
        __asm__ volatile("vfmacc.vf      v30,  %0, v20": : "f"(-FPCONST(2.0))); //                                          u3p[2]-FPCONST(2.0)*u3p[1]        ;
        __asm__ volatile("vfadd.vv       v19, v30, v19");                       //                                                                    +u3p[0] ;
        __asm__ volatile("vmv.v.v        v30, v20");
        __asm__ volatile("vfmacc.vf      v30,  %0, v24": : "f"(a31));           // u3p[1]+a31*du11                                                            ;
        __asm__ volatile("vfmacc.vf      v30,  %0, v25": : "f"(a32));           //                +a32*du21                                                   ;
        __asm__ volatile("vfmacc.vf      v30,  %0, v26": : "f"(a33));           //                         +a33*du31                                          ;
        __asm__ volatile("vfmacc.vf      v30,  %0, v19": : "f"(sig));           //                                   + sig*(                                 );

        __asm__ volatile("vfmacc.vf      v16,  %0, v21": : "f"(-FPCONST(2.0))); //                                          u3p[3]-FPCONST(2.0)*u3p[2]        ;
        __asm__ volatile("vfadd.vv       v16, v16, v20");                       //                                                                    +u3p[1] ;
        __asm__ volatile("vmv.v.v        v31, v21");
        __asm__ volatile("vfmacc.vf      v31,  %0, v27": : "f"(a31));           // u3p[2]+a31*du12                                                            ;
        __asm__ volatile("vfmacc.vf      v31,  %0, v28": : "f"(a32));           //                +a32*du22                                                   ;
        __asm__ volatile("vfmacc.vf      v31,  %0, v29": : "f"(a33));           //                         +a33*du32                                          ;
        __asm__ volatile("vfmacc.vf      v31,  %0, v16": : "f"(sig));           //                                   + sig*(                                 );

        __asm__ volatile("vssseg2e32.v   v30, (%0), %1": : "r"(&u3[ioutb+1]), "r"(3*sizeof(float)));

        ky += vl;
      }
      /* for ( ky=1 ; ky<n ; ky++ ) { */
      /*   int curb = (ky)*3; */
      /*   int ioutb = nl2*n*3+ky*3; */
      /*   float du11, du21, du31, du12, du22, du32; */

      /*   float* u1p = u1 + curb; */
      /*   float* u2p = u2 + curb; */
      /*   float* u3p = u3 + curb; */
      /*   // (-2 -1) (0 1 2) (3 4 5) */
      /*   du11 = u1p[4] - u1p[-2]; */
      /*   du21 = u2p[4] - u2p[-2]; */
      /*   du31 = u3p[4] - u3p[-2]; */
      /*   du12 = u1p[5] - u1p[-1]; */
      /*   du22 = u2p[5] - u2p[-1]; */
      /*   du32 = u3p[5] - u3p[-1]; */

      /*   float u11 = u1p[1]+a11*du11+a12*du21+a13*du31 + sig*(u1p[2]-FPCONST(2.0)*u1p[1]+u1p[0]); */
      /*   float u21 = u2p[1]+a21*du11+a22*du21+a23*du31 + sig*(u2p[2]-FPCONST(2.0)*u2p[1]+u2p[0]); */
      /*   float u31 = u3p[1]+a31*du11+a32*du21+a33*du31 + sig*(u3p[2]-FPCONST(2.0)*u3p[1]+u3p[0]); */

      /*   float u12 = u1p[2]+a11*du12+a12*du22+a13*du32 + sig*(u1p[3]-FPCONST(2.0)*u1p[2]+u1p[1]); */
      /*   float u22 = u2p[2]+a21*du12+a22*du22+a23*du32 + sig*(u2p[3]-FPCONST(2.0)*u2p[2]+u2p[1]); */
      /*   float u32 = u3p[2]+a31*du12+a32*du22+a33*du32 + sig*(u3p[3]-FPCONST(2.0)*u3p[2]+u3p[1]); */

        /* printf("u1p %x %x %x %x %x %x %x %x\n", */
        /*        hex(u1p[-2]), hex(u1p[-1]), hex(u1p[0]), hex(u1p[1]), hex(u1p[2]), hex(u1p[3]), hex(u1p[4]), hex(u1p[5])); */
        /* printf("u2p %x %x %x %x %x %x %x %x\n", */
        /*        hex(u2p[-2]), hex(u2p[-1]), hex(u2p[0]), hex(u2p[1]), hex(u2p[2]), hex(u2p[3]), hex(u2p[4]), hex(u2p[5])); */
        /* printf("u3p %x %x %x %x %x %x %x %x\n", */
        /*        hex(u3p[-2]), hex(u3p[-1]), hex(u3p[0]), hex(u3p[1]), hex(u3p[2]), hex(u3p[3]), hex(u3p[4]), hex(u3p[5])); */
        /* printf("du %x %x %x %x %x %x\n", hex(du11), hex(du21), hex(du31), hex(du12), hex(du22), hex(du32)); */

        /* printf("%x %x %x %x %x %x\n", */
        /*        hex(u1p[2]-FPCONST(2.0)*u1p[1]), */
        /*        hex(u1p[2]-FPCONST(2.0)*u1p[1]+u1p[0]), */
        /*        hex(u1p[1]+a11*du11), */
        /*        hex(u1p[1]+a11*du11+a12*du21), */
        /*        hex(u1p[1]+a11*du11+a12*du21+a13*du31), */
        /*        hex(u1p[1]+a11*du11+a12*du21+a13*du31+sig*(u1p[2]-FPCONST(2.0)*u1p[1]+u1p[0]))); */

        /* printf("%x %x\n", hex(u11), hex(u1[(ioutb+1)])); */
        /* printf("%x %x\n", hex(u21), hex(u2[(ioutb+1)])); */
        /* printf("%x %x\n", hex(u31), hex(u3[(ioutb+1)])); */
        /* printf("%x %x\n", hex(u12), hex(u1[(ioutb+2)])); */
        /* printf("%x %x\n", hex(u22), hex(u2[(ioutb+2)])); */
        /* printf("%x %x\n", hex(u32), hex(u3[(ioutb+2)])); */

        /* if (u11 != u1[(ioutb+1)]) exit(1); */
        /* if (u21 != u2[(ioutb+1)]) exit(1); */
        /* if (u31 != u3[(ioutb+1)]) exit(1); */
        /* if (u12 != u1[(ioutb+2)]) exit(1); */
        /* if (u22 != u2[(ioutb+2)]) exit(1); */
        /* if (u32 != u3[(ioutb+2)]) exit(1); */

      /* } */

#else
      for ( ky=1 ; ky<n ; ky++ ) {
        int curb = (ky)*3;
        int ioutb = nl2*n*3+ky*3;
        float du11, du21, du31, du12, du22, du32;

        float* u1p = u1 + curb;
        float* u2p = u2 + curb;
        float* u3p = u3 + curb;
        // (-2 -1) (0 1 2) (3 4 5)
        du11 = u1p[4] - u1p[-2];
        du21 = u2p[4] - u2p[-2];
        du31 = u3p[4] - u3p[-2];
        du12 = u1p[5] - u1p[-1];
        du22 = u2p[5] - u2p[-1];
        du32 = u3p[5] - u3p[-1];

        u1[(ioutb+1)]= u1p[1]+a11*du11+a12*du21+a13*du31 + sig*(u1p[2]-FPCONST(2.0)*u1p[1]+u1p[0]);
        u2[(ioutb+1)]= u2p[1]+a21*du11+a22*du21+a23*du31 + sig*(u2p[2]-FPCONST(2.0)*u2p[1]+u2p[0]);
        u3[(ioutb+1)]= u3p[1]+a31*du11+a32*du21+a33*du31 + sig*(u3p[2]-FPCONST(2.0)*u3p[1]+u3p[0]);

        u1[(ioutb+2)]= u1p[2]+a11*du12+a12*du22+a13*du32 + sig*(u1p[3]-FPCONST(2.0)*u1p[2]+u1p[1]);
        u2[(ioutb+2)]= u2p[2]+a21*du12+a22*du22+a23*du32 + sig*(u2p[3]-FPCONST(2.0)*u2p[2]+u2p[1]);
        u3[(ioutb+2)]= u3p[2]+a31*du12+a32*du22+a33*du32 + sig*(u3p[3]-FPCONST(2.0)*u3p[2]+u3p[1]);

      }

      /* for ( kx=1 ; kx<3 ; kx++ ){ */
      /*   for ( ky=1 ; ky<n ; ky++ ) { */
      /*     int next=(ky+1)*3+kx; */
      /*     int prev=(ky-1)*3+kx; */
      /*     int cur=(ky)*3+kx; */
      /*     int iout=nl2*n*3+ky*3+kx; */
      /*     du1[ky] = u1[next] - u1[prev]; */
      /*     du2[ky] = u2[next] - u2[prev]; */
      /*     du3[ky] = u3[next] - u3[prev]; */
      /*     u1[iout]= */
      /*       u1[cur]+a11*du1[ky]+a12*du2[ky]+a13*du3[ky] + sig* */
      /*       (u1[cur+1]-FPCONST(2.0)*u1[cur]+u1[cur-1]); */
      /*     u2[iout]= */
      /*       u2[cur]+a21*du1[ky]+a22*du2[ky]+a23*du3[ky] + sig* */
      /*       (u2[cur+1]-FPCONST(2.0)*u2[cur]+u2[cur-1]); */
      /*     u3[iout]= */
      /*       u3[cur]+a31*du1[ky]+a32*du2[ky]+a33*du3[ky] + sig* */
      /*       (u3[cur+1]-FPCONST(2.0)*u3[cur]+u3[cur-1]); */
      /*   } */
      /* } */
#endif
        ret+=get_array_feedback(&u1[3*n],3*n)+get_array_feedback(&u2[3*n],3*n)+get_array_feedback(&u3[3*n],3*n);
    }
    return ret;
}

e_fp integrate_predictors(loops_params *p) {
	e_fp *dm=p->v[0];
	e_fp *px=p->m3[0];
	int n=p->N, loop=p->Loop;
	int l,i;
	e_fp ret=FPCONST(0.0);
	reinit_vec(p,dm,30);
	reinit_vec(p,px,13*n);
    /*
     *******************************************************************
     *   Kernel 9 -- integrate predictors
     *******************************************************************
     *    DO 9  L = 1,Loop
     *    DO 9  i = 1,n
     *    PX( 1,i)= DM28*PX(13,i) + DM27*PX(12,i) + DM26*PX(11,i) +
     *   .          DM25*PX(10,i) + DM24*PX( 9,i) + DM23*PX( 8,i) +
     *   .          DM22*PX( 7,i) +  C0*(PX( 5,i) +      PX( 6,i))+ PX( 3,i)
     *  9 CONTINUE
     */

    for ( l=1 ; l<=loop ; l++ ) {
		reinit_vec(p,dm,30); /* reinit to make sure each loop needs to be executed */
        for ( i=0 ; i<n ; i++ ) {
            px[i*13] = dm[28]*px[i*13+12] + dm[27]*px[i*13+11] + dm[26]*px[i*13+10] +
                       dm[25]*px[i*13+ 9] + dm[24]*px[i*13+ 8] + dm[23]*px[i*13+ 7] +
                       dm[22]*px[i*13+ 6] + dm[0]*( px[i*13+ 4] + px[i*13+ 5]) + px[i*13+ 2];
        }
		for ( i=0 ; i<n ; i++ )
			ret+=px[i*13];
		ret/=(e_fp)n; /* feedback to make sure all loops get executed */
#if (BMDEBUG && DEBUG_ACCURATE_BITS)
	th_printf("\nipred %d:",debug_counter++);
	th_print_fp(ret);
#endif
    }
	return ret;
}

e_fp difference_predictors(loops_params *p) {
	e_fp ar,br,cr;
	e_fp *px=p->m3[0];
	e_fp *cx=p->v[0];
	int n=p->N, loop=p->Loop;
	int i,l;
	reinit_vec(p,px,13*n);
	reinit_vec(p,cx,n);
    /*
     *******************************************************************
     *   Kernel 10 -- difference predictors
     *******************************************************************
     *    DO 10  L= 1,Loop
     *    DO 10  i= 1,n
     *    AR      =      CX(5,i)
     *    BR      = AR - PX(5,i)
     *    PX(5,i) = AR
     *    CR      = BR - PX(6,i)
     *    PX(6,i) = BR
     *    AR      = CR - PX(7,i)
     *    PX(7,i) = CR
     *    BR      = AR - PX(8,i)
     *    PX(8,i) = AR
     *    CR      = BR - PX(9,i)
     *    PX(9,i) = BR
     *    AR      = CR - PX(10,i)
     *    PX(10,i)= CR
     *    BR      = AR - PX(11,i)
     *    PX(11,i)= AR
     *    CR      = BR - PX(12,i)
     *    PX(12,i)= BR
     *    PX(14,i)= CR - PX(13,i)
     *    PX(13,i)= CR
     * 10 CONTINUE
     */

    for ( l=1 ; l<=loop ; l++ ) {
		reinit_vec(p,cx,n); /* initializers to make sure all loops need to be executed */
        for ( i=0 ; i<n ; i++ ) {
            ar        =      cx[i];
            br        = ar - px[14*i+ 4];
            px[14*i+ 4] = ar;
            cr        = br - px[14*i+ 5];
            px[14*i+ 5] = br;
            ar        = cr - px[14*i+ 6];
            px[14*i+ 6] = cr;
            br        = ar - px[14*i+ 7];
            px[14*i+ 7] = ar;
            cr        = br - px[14*i+ 8];
            px[14*i+ 8] = br;
            ar        = cr - px[14*i+ 9];
            px[14*i+ 9] = cr;
            br        = ar - px[14*i+10];
            px[14*i+10] = ar;
            cr        = br - px[14*i+11];
            px[14*i+11] = br;
            px[14*i+13] = cr - px[14*i+12];
            px[14*i+12] = cr;
        }
#if (BMDEBUG && DEBUG_ACCURATE_BITS)
	th_printf("\npred %d:",debug_counter++);
	th_print_fp(px[0]);
#endif
    }
	return px[0];
}

e_fp first_sum(loops_params *p) {
	e_fp *x=p->v[0];
	e_fp *y=p->v[1];
	int n=p->N, loop=p->Loop;
	int k,l;
	e_fp ret=FPCONST(0.0);
	reinit_vec(p,y,n);
    /*
     *******************************************************************
     *   Kernel 11 -- first sum
     *******************************************************************
     *    DO 11 L = 1,Loop
     *        X(1)= Y(1)
     *    DO 11 k = 2,n
     * 11     X(k)= X(k-1) + Y(k)
     */

    for ( l=1 ; l<=loop ; l++ ) {
		reinit_vec(p,y,n);
        x[0] = y[0];
        for ( k=1 ; k<n ; k++ ) {
            x[k] = x[k-1] + y[k];
        }
		ret+=x[n-1];
#if (BMDEBUG && DEBUG_ACCURATE_BITS)
	th_printf("\nsum %d:",debug_counter++);
	th_print_fp(ret);
#endif
    }
	ret/=loop;
	return ret;
}

e_fp first_dif(loops_params *p) {
	e_fp *x=p->v[0];
	e_fp *y=p->v[1];
	int n=p->N, loop=p->Loop;
	int k,l;
    /*
     *******************************************************************
     *   Kernel 12 -- first difference
     *******************************************************************
     *    DO 12 L = 1,Loop
     *    DO 12 k = 1,n
     * 12     X(k)= Y(k+1) - Y(k)
     */

	reinit_vec(p,x,n);
    for ( l=0 ; l<=loop ; l++ ) {
		reinit_vec(p,y,n+1);
        for ( k=0 ; k<n ; k++ ) {
            x[k]+= y[k+1] - y[k];
        }
    }

	return get_array_feedback(x,n);
}

e_fp pic_2d(loops_params *p) {
	e_fp *y=p->v[0],  *z=p->v[1];
	e_fp *p1r=p->v[2],  *p1i=p->v[3];
	e_fp *p2r=p->v[4],  *p2i=p->v[5];
	e_u32 *e=p->iv[0],  *f=p->iv[1];
	e_fp *b=p->m2[1],  *c=p->m2[2],  *h=p->m2[3];
	int n=p->N, loop=p->Loop,l,ip;
	int i1=0,j1=0,i2=0,j2=0;

	if (p->m2size < 64*64) {
		p->err="pic_2d bad size for matrix";
		return FPCONST(0.0);
	}
	if ((p->vsize < (64+32)) || (p->ivsize < (64+32))) {
		p->err="pic_2d bad size for vectors";
		return FPCONST(0.0);
	}
   /*
     *******************************************************************
     *   Kernel 13 -- 2-D PIC (Particle In Cell)
     *******************************************************************
     *    DO  13     L= 1,Loop
     *    DO  13    ip= 1,n
     *              i1= P(1,ip)
     *              j1= P(2,ip)
     *              i1=        1 + MOD2N(i1,64)
     *              j1=        1 + MOD2N(j1,64)
     *         P(3,ip)= P(3,ip)  + B(i1,j1)
     *         P(4,ip)= P(4,ip)  + C(i1,j1)
     *         P(1,ip)= P(1,ip)  + P(3,ip)
     *         P(2,ip)= P(2,ip)  + P(4,ip)
     *              i2= P(1,ip)
     *              j2= P(2,ip)
     *              i2=            MOD2N(i2,64)
     *              j2=            MOD2N(j2,64)
     *         P(1,ip)= P(1,ip)  + Y(i2+32)
     *         P(2,ip)= P(2,ip)  + Z(j2+32)
     *              i2= i2       + E(i2+32)
     *              j2= j2       + F(j2+32)
     *        H(i2,j2)= H(i2,j2) + 1.0
     * 13 CONTINUE
     */
	reinit_vec_const(p,p1r,n,FPCONST(1.0));
	reinit_vec_const(p,p1i,n,FPCONST(0.5));
	reinit_vec_const(p,p2r,n,FPCONST(1.0));
	reinit_vec_const(p,p2i,n,FPCONST(0.5));
	reinit_vec(p,b,64*64+4);
	reinit_vec(p,c,64*64+4);
	reinit_vec(p,h,64*64+4);
	reinit_vec(p,y,100);
	reinit_vec(p,z,100);
	reinit_ivec(p,e,100,1);
	reinit_ivec(p,f,100,1);

    for ( l=1 ; l<=loop ; l++ ) {
#if USE_RVV
      for ( ip=0 ; ip<n ; ) {
        size_t vl = __riscv_vsetvl_e32m2(n - ip);
        vfloat32m2_t p1rs = __riscv_vle32_v_f32m2(&p1r[ip], vl);
        vfloat32m2_t p1is = __riscv_vle32_v_f32m2(&p1i[ip], vl);
        vint32m2_t i1s = __riscv_vfcvt_rtz_x_f_v_i32m2(p1rs, vl);
        vint32m2_t j1s = __riscv_vfcvt_rtz_x_f_v_i32m2(p1is, vl);

        i1s = __riscv_vand_vx_i32m2(i1s, 0x03f, vl);
        j1s = __riscv_vand_vx_i32m2(i1s, 0x03f, vl);

        vint32m2_t bcidxs = __riscv_vsll_vx_i32m2(j1s, 6, vl);
        bcidxs = __riscv_vadd_vv_i32m2(bcidxs, i1s, vl);
        bcidxs = __riscv_vsll_vx_i32m2(bcidxs, 2, vl);
        vuint64m4_t bcidxs64 = __riscv_vreinterpret_v_i64m4_u64m4(__riscv_vsext_vf2_i64m4(bcidxs, vl));

        vfloat32m2_t bs = __riscv_vluxei64_v_f32m2(b, bcidxs64, vl);
        vfloat32m2_t cs = __riscv_vluxei64_v_f32m2(c, bcidxs64, vl);
        vfloat32m2_t p2rs = __riscv_vle32_v_f32m2(&p2r[ip], vl);
        vfloat32m2_t p2is = __riscv_vle32_v_f32m2(&p2i[ip], vl);
        p2rs = __riscv_vfadd_vv_f32m2(p2rs, bs, vl);
        p2is = __riscv_vfadd_vv_f32m2(p2is, cs, vl);
        p1rs = __riscv_vfadd_vv_f32m2(p2rs, p1rs, vl);
        p1is = __riscv_vfadd_vv_f32m2(p2is, p1is, vl);

        vint32m2_t i2s = __riscv_vfcvt_rtz_x_f_v_i32m2(p1rs, vl);
        vint32m2_t j2s = __riscv_vfcvt_rtz_x_f_v_i32m2(p1is, vl);

        i2s = __riscv_vand_vx_i32m2(i2s, 0x03f, vl);
        j2s = __riscv_vand_vx_i32m2(i2s, 0x03f, vl);

        vint32m2_t yeidxs = __riscv_vadd_vx_i32m2(i2s, 32, vl);
        yeidxs = __riscv_vsll_vx_i32m2(i2s, 2, vl);
        vuint64m4_t yeidxs64 = __riscv_vreinterpret_v_i64m4_u64m4(__riscv_vsext_vf2_i64m4(yeidxs, vl));
        vint32m2_t zfidxs = __riscv_vadd_vx_i32m2(j2s, 32, vl);
        zfidxs = __riscv_vsll_vx_i32m2(j2s, 2, vl);
        vuint64m4_t zfidxs64 = __riscv_vreinterpret_v_i64m4_u64m4(__riscv_vsext_vf2_i64m4(zfidxs, vl));

        vfloat32m2_t ys = __riscv_vluxei64_v_f32m2(y, yeidxs64, vl);
        vuint32m2_t es = __riscv_vluxei64_v_u32m2(e, yeidxs64, vl);
        vfloat32m2_t zs = __riscv_vluxei64_v_f32m2(z, zfidxs64, vl);
        vuint32m2_t fs = __riscv_vluxei64_v_u32m2(f, zfidxs64, vl);
        p1rs = __riscv_vfadd_vv_f32m2(p1rs, ys, vl);
        p1is = __riscv_vfadd_vv_f32m2(p1is, zs, vl);

        __riscv_vse32_v_f32m2(&p2r[ip], p2rs, vl);
        __riscv_vse32_v_f32m2(&p2i[ip], p2is, vl);
        __riscv_vse32_v_f32m2(&p1r[ip], p1rs, vl);
        __riscv_vse32_v_f32m2(&p1i[ip], p1is, vl);

        i2s = __riscv_vadd_vv_i32m2(i2s, __riscv_vreinterpret_v_u32m2_i32m2(es), vl);
        j2s = __riscv_vadd_vv_i32m2(j2s, __riscv_vreinterpret_v_u32m2_i32m2(fs), vl);

        vint32m2_t hidxs = __riscv_vsll_vx_i32m2(j2s, 6, vl);
        hidxs = __riscv_vadd_vv_i32m2(hidxs, i2s, vl);

        for (size_t j = 0; j < vl; j++) {
          int32_t idx = __riscv_vmv_x_s_i32m2_i32(hidxs);
          hidxs = __riscv_vslide1down_vx_i32m2(hidxs, 0, vl);
          h[idx] += 1.0;
        }
        ip += vl;
      }

#else
      for ( ip=0 ; ip<n ; ip++ ) {
        i1 = p1r[ip];
        j1 = p1i[ip];
        i1 &= 0x03f;
        j1 &= 0x03f;
        p2r[ip] += b[j1*64+i1];
        p2i[ip] += c[j1*64+i1];
        p1r[ip] += p2r[ip];
        p1i[ip] += p2i[ip];
        i2 = p1r[ip];
        j2 = p1i[ip];
        i2 &=0x03f;
        j2 &=0x03f;
        p1r[ip] += y[i2+32];
        p1i[ip] += z[j2+32];
        i2 += e[i2+32];
        j2 += f[j2+32];
        h[j2*64+i2] += 1.0;
      }
#endif
#if (BMDEBUG && DEBUG_ACCURATE_BITS)
	th_printf("\n2d %d:",debug_counter++);
	th_print_fp(h[0]+p1r[0]+p1i[0]+p2r[0]+p2i[0]);
#endif
    }
	return h[0]+p1r[0]+p1i[0]+p2r[0]+p2i[0];
}
int is_2px(int x)
{
    return (x & (x - 1)) == 0;
}
e_fp pic_1d(loops_params *p) {
	e_fp *vx=p->v[0],  *xx=p->v[1];
	e_fp *grd=p->v[2],  *xi=p->v[3];
	e_fp *ex=p->v[4],  *ex1=p->v[5];
	e_fp *dex=p->v[6],  *dex1=p->v[7];
	e_fp *rx=p->v[8];
	e_fp *rh=p->v[9];
	e_u32 *ix=p->iv[0],  *ir=p->iv[1]; /* temporary outputs */
	e_fp dw,flx;
	int n=p->N, loop=p->Loop;
	int k,l;
    /*
     *******************************************************************
     *   Kernel 14 -- 1-D PIC (pticle In Cell)
     *******************************************************************
     *    DO   14   L= 1,Loop
     *    DO   141  k= 1,n
     *          VX(k)= 0.0
     *          XX(k)= 0.0
     *          IX(k)= INT(  GRD(k))
     *          XI(k)= e_fp( IX(k))
     *         EX1(k)= EX   ( IX(k))
     *        DEX1(k)= DEX  ( IX(k))
     *41  CONTINUE
     *    DO   142  k= 1,n
     *          VX(k)= VX(k) + EX1(k) + (XX(k) - XI(k))*DEX1(k)
     *          XX(k)= XX(k) + VX(k)  + FLX
     *          IR(k)= XX(k)
     *          RX(k)= XX(k) - IR(k)
     *          IR(k)= MOD2N(  IR(k),2048) + 1
     *          XX(k)= RX(k) + IR(k)
     *42  CONTINUE
     *    DO  14    k= 1,n
     *    RH(IR(k)  )= RH(IR(k)  ) + 1.0 - RX(k)
     *    RH(IR(k)+1)= RH(IR(k)+1) + RX(k)
     *14  CONTINUE
     */
	if (!is_2px(n)) {
		th_exit(TH_ERROR,"Input for pic_1d in loops must be power of 2 [got %x].\n",n);
	}
	reinit_ivec_pos(p, ix, n, n-1); /* n 1..n numbers */
	reinit_vec(p, dex, n); /* n 1..n numbers */
	dw= FPCONST(-100.000e0);
    for ( k=0 ; k<n ; k++ ) {
		dex[k] =  dw*dex[k];
		grd[k] = ix[k];
	}
	flx=FPCONST(0.00100e0);

    for ( l=1 ; l<=loop ; l++ ) {
        for ( k=0 ; k<n ; k++ ) {
            vx[k] = FPCONST(0.0);
            xx[k] = FPCONST(0.0);
            ix[k] = (long) grd[k];
            xi[k] = (e_fp) ix[k];
            ex1[k] = ex[ ix[k] - 1 ];
            dex1[k] = dex[ ix[k] - 1 ];
			/* add an epsilon to make sure this loop needs to be executed
			   each time through the outer loop */
			grd[k]+=EE_EPSINI;
        }
        for ( k=0 ; k<n ; k++ ) {
            vx[k] += ex1[k] + ( xx[k] - xi[k] )*dex1[k];
            xx[k] += vx[k]  + flx;
            ir[k] = (e_u32)xx[k];
            rx[k] = xx[k] - ir[k];
            ir[k] = ( ir[k] & (n-1) ) + 1;
            xx[k] = rx[k] + ir[k];
        }
        for ( k=0 ; k<n ; k++ ) {
            rh[ ir[k]-1 ] += FPCONST(1.0) - rx[k];
            rh[ ir[k]   ] += rx[k];
        }
#if (BMDEBUG && DEBUG_ACCURATE_BITS)
	th_printf("\n1d %d:",debug_counter++);
	th_print_fp(rh[0]);
#endif
    }
	return rh[0];
}

e_fp casual(loops_params *p) {
	e_fp *vy=p->m2[0];
	e_fp *vh=p->m2[1];
	e_fp *vf=p->m2[2];
	e_fp *vg=p->m2[3];
	e_fp *vs=p->m2[4];
	e_fp r=FPCONST(0.0),t=FPCONST(0.0),s=FPCONST(0.0);
	int n=p->N, loop=p->Loop;
	int j,k,l;
	e_fp test=FPCONST(0.0);

    /*
     *******************************************************************
     *   Kernel 15 -- Casual Fortran.  Development version
     *******************************************************************
     *      DO 45  L = 1,Loop
     *             NG= 7
     *             NZ= n
     *             AR= 0.053
     *             BR= 0.073
     * 15   DO 45  j = 2,NG
     *      DO 45  k = 2,NZ
     *             IF( j-NG) 31,30,30
     * 30     VY(k,j)= 0.0
     *                 GO TO 45
     * 31          IF( VH(k,j+1) -VH(k,j)) 33,33,32
     * 32           T= AR
     *                 GO TO 34
     * 33           T= BR
     * 34          IF( VF(k,j) -VF(k-1,j)) 35,36,36
     * 35           R= MAX( VH(k-1,j), VH(k-1,j+1))
     *              S= VF(k-1,j)
     *                 GO TO 37
     * 36           R= MAX( VH(k,j),   VH(k,j+1))
     *              S= VF(k,j)
     * 37     VY(k,j)= SQRT( VG(k,j)**2 +R*R)*T/S
     * 38          IF( k-NZ) 40,39,39
     * 39     VS(k,j)= 0.
     *                 GO TO 45
     * 40          IF( VF(k,j) -VF(k,j-1)) 41,42,42
     * 41           R= MAX( VG(k,j-1), VG(k+1,j-1))
     *              S= VF(k,j-1)
     *              T= BR
     *                 GO TO 43
     * 42           R= MAX( VG(k,j),   VG(k+1,j))
     *              S= VF(k,j)
     *              T= AR
     * 43     VS(k,j)= SQRT( VH(k,j)**2 +R*R)*T/S
     * 45    CONTINUE
     */


    for ( l=1 ; l<=loop ; l++ ) {
        int ng = 7;
        int nz = n;
        e_fp ar = FPCONST(0.053);
        e_fp br = FPCONST(0.073);
		/* vh, vf and vg must be modified at each loop, otherwise compiler can optimize away */
		reinit_vec(p,vh,nz*7);
		reinit_vec(p,vf,nz*7);
		reinit_vec(p,vg,nz*7);
        for ( j=1 ; j<ng ; j++ ) {
            for ( k=1 ; k<nz ; k++ ) {
				int cur=j*nz+k;
				int down=(j+1)*nz+k;
				int up=(j-1)*nz+k;
				int left=cur-1;
                if ( (j+1) >= ng ) {
                    vy[cur] = FPCONST(0.0);
                    continue;
                }
                if ( vh[down] > vh[cur] ) {
                    t = ar;
                }
                else {
                    t = br;
                }
                if ( vf[cur] < vf[left] ) {
                    if ( vh[left] > vh[down-1] )
                        r = vh[left];
                    else
                        r = vh[down-1];
                    s = vf[left];
                }
                else {
                    if ( vh[cur] > vh[down] )
                        r = vh[cur];
                    else
                        r = vh[down];
                    s = vf[cur];
                }
                vy[cur] = th_sqrt( vg[cur]*vg[cur] + r*r )* t/s;
                if ( (k+1) >= nz ) {
                    vs[cur] = FPCONST(0.0);
                    continue;
                }
                if ( vf[cur] < vf[up] ) {
                    if ( vg[up] > vg[up+1] )
                        r = vg[up];
                    else
                        r = vg[up+1];
                    s = vf[up];
                    t = br;
                }
                else {
                    if ( vg[cur] > vg[cur+1] )
                        r = vg[cur];
                    else
                        r = vg[cur+1];
                    s = vf[cur];
                    t = ar;
                }
                vs[cur] = th_sqrt( vh[cur]*vh[cur] + r*r )* t / s;
            }
        }
		/* Save some value of the output so compiler does not optimize the compute workload away */
		test+=get_array_feedback(vs,nz*ng);
		test+=get_array_feedback(vy,nz*ng);
   }
	return test;
}

e_fp monte_carlo(loops_params *p) {
	e_fp *d=p->v[0];
	e_u32 *zone=p->iv[1];
	e_fp *plan=p->v[2],tmp;
	e_fp *tmps=p->v[3];
	e_fp r,s,t;
	e_u32 m;

	int n=p->N, loop=p->Loop;
	int k,l;
	int ret=0;
    /*
     *******************************************************************
     *   Kernel 16 -- Monte Carlo search loop
     *******************************************************************
     *          II= n/3
     *          LB= II+II
     *          k2= 0
     *          k3= 0
     *    DO 485 L= 1,Loop
     *           m= 1
     *405       i1= m
     *410       j2= (n+n)*(m-1)+1
     *    DO 470 k= 1,n
     *          k2= k2+1
     *          j4= j2+k+k
     *          j5= ZONE(j4)
     *          IF( j5-n      ) 420,475,450
     *415       IF( j5-n+II   ) 430,425,425
     *420       IF( j5-n+LB   ) 435,415,415
     *425       IF( PLAN(j5)-R) 445,480,440
     *430       IF( PLAN(j5)-S) 445,480,440
     *435       IF( PLAN(j5)-T) 445,480,440
     *440       IF( ZONE(j4-1)) 455,485,470
     *445       IF( ZONE(j4-1)) 470,485,455
     *450       k3= k3+1
     *          IF( D(j5)-(D(j5-1)*(T-D(j5-2))**2+(S-D(j5-3))**2
     *   .                        +(R-D(j5-4))**2)) 445,480,440
     *455        m= m+1
     *          IF( m-ZONE(1) ) 465,465,460
     *460        m= 1
     *465       IF( i1-m) 410,480,410
     *470 CONTINUE
     *475 CONTINUE
     *480 CONTINUE
     *485 CONTINUE
     */

    int ii = n / 3;
    int lb = ii + ii;
    int k3=0, k2 = 0;
	int maxvec=64;
	int imask=maxvec-1;
	reinit_vec(p,plan,maxvec);
	reinit_vec(p,d,maxvec);
    for ( l=1 ; l<=loop ; l++ ) {
		int i1,j2,j4,j5;
        i1 = m = 1;
		/* inits to make sure new inputs each loop iteration so compiler does not optimize the work away */
		reinit_ivec(p,zone,(2*n)*(n+1),imask);
		reinit_vec(p,tmps,4);
		r=tmps[0];
		s=tmps[1];
		t=tmps[2];
        label410:
        j2 = ( n + n )*( m - 1 ) + 1;
        for ( k=1 ; k<=n ; k++ ) {
            k2++;
            j4 = j2 + k + k;
            j5 = zone[j4-1]+1;
            if ( j5 < n ) {
                if ( j5+lb < n ) {              /* 420 */
                    tmp = plan[j5-1] - t;       /* 435 */
                } else {
                    if ( j5+ii < n ) {          /* 415 */
                        tmp = plan[j5-1] - s;   /* 430 */
                    } else {
                        tmp = plan[j5-1] - r;   /* 425 */
                    }
                }
            } else if( j5 == n ) {
                break;                          /* 475 */
            } else {
                k3++;                           /* 450 */
                tmp=(d[j5-1]-(d[j5-2]*(t-d[j5-3])*(t-d[j5-3])+(s-d[j5-4])*
                              (s-d[j5-4])+(r-d[j5-5])*(r-d[j5-5])));
            }
            if ( tmp < FPCONST(0.0) ) {
                if ( zone[j4-2] > 0 )           /* 445 */
                    continue;                   /* 470 */
                else if ( !zone[j4-2] )
                    break;                      /* 480 */
            } else if ( tmp != FPCONST(0.0) ) {
                if ( zone[j4-2] > 0 )           /* 440 */
                    continue;                   /* 470 */
                else if ( !zone[j4-2] )
                    break;                      /* 480 */
            } else break;                       /* 485 */
            m++;                                /* 455 */
            if ( m > zone[0] )
                m = 1;                          /* 460 */
            if (( i1-m ) != 0)                  /* 465 */
                goto label410;
            else
                break;
        }
		/* return value should have an update output for wach loop iteration so it does not get optimized away */
		ret+=m;
    }
	return (e_fp)ret;
}

e_fp implicit(loops_params *p) {
	e_fp *vxne=p->v[0],  *vxnd=p->v[1];
	e_fp *vlr=p->v[2],  *vlin=p->v[3];
	e_fp *vsp=p->v[4],  *vstp=p->v[5];
	e_fp *ve3=p->v[6];
    e_fp    start_scale = FPCONST(5.0) / FPCONST(3.0);
    e_fp    start_xnm = FPCONST(1.0) / FPCONST(3.0);
    e_fp    start_e6 = FPCONST(1.03) / FPCONST(3.07);
	e_fp	ret=FPCONST(0.0),e3,xnei,xnc;
	int n=p->N, loop=p->Loop,l;
	e_fp	bignum=EE_MININI;

    /*
     *******************************************************************
     *   Kernel 17 -- implicit, conditional computation
     *******************************************************************
     *          DO 62 L= 1,Loop
     *                i= n
     *                j= 1
     *              INK= -1
     *            SCALE= 5./3.
     *              XNM= 1./3.
     *               E6= 1.03/3.07
     *                   GO TO 61
     *60             E6= XNM*VSP(i)+VSTP(i)
     *          VXNE(i)= E6
     *              XNM= E6
     *           VE3(i)= E6
     *                i= i+INK
     *               IF( i.EQ.j) GO TO  62
     *61             E3= XNM*VLR(i) +VLIN(i)
     *             XNEI= VXNE(i)
     *          VXND(i)= E6
     *              XNC= SCALE*E3
     *               IF( XNM .GT.XNC) GO TO  60
     *               IF( XNEI.GT.XNC) GO TO  60
     *           VE3(i)= E3
     *               E6= E3+E3-XNM
     *          VXNE(i)= E3+E3-XNEI
     *              XNM= E6
     *                i= i+INK
     *               IF( i.NE.j) GO TO 61
     * 62 CONTINUE
     */

	 reinit_vec(p,vlin,n);
	 reinit_vec(p,vlr,n);
	 reinit_vec(p,vsp,n);
	 reinit_vec(p,vstp,n);
	 reinit_vec(p,vxne,n);

    for ( l=1 ; l<=loop ; l++ ) {
        int i = n-1;
        int j = 0;
        int ink = -1;
        e_fp scale = start_scale;
        e_fp xnm = start_xnm;
        e_fp e6 = start_e6;
		/* add an epsilon every iteration otherwise compiler only needs to execute last iteration */
		start_xnm+=FPCONST(1e-20);
		start_e6+=FPCONST(1e-20);
        goto l61;
l60:    e6 = xnm*vsp[i] + vstp[i];
		if (!th_isfinite(e6)) {e3=start_xnm; e6=start_e6; xnm=start_xnm;};
        vxne[i] = e6;
        xnm = e6;
        ve3[i] = e6;
        i += ink;
        if ( i==j ) goto l62;
l61:    e3 = xnm*vlr[i] + vlin[i];
		if (!th_isfinite(e3) || e3>bignum) {e3=start_xnm; e6=start_e6; xnm=start_xnm;}
        xnei = vxne[i];
        vxnd[i] = e6;
        xnc = scale*e3;
        if ( xnm > xnc ) goto l60;
        if ( xnei > xnc ) goto l60;
        ve3[i] = e3;
        e6 = e3  - xnm + e3;
        vxne[i] = e3  - xnei + e3;
        xnm = e6;
        i += ink;
        if ( i != j ) goto l61;
l62:;
		ret+=get_array_feedback(vxnd,n);
		ret+=get_array_feedback(vxne,n);
    }
	return ret;
}

e_fp hydro_2d(loops_params *p) {
	e_fp *za=p->m2[0];
	e_fp *zb=p->m2[1];
	e_fp *zp=p->m2[2];
	e_fp *zq=p->m2[3];
	e_fp *zm=p->m2[4];
	e_fp *zr=p->m2[5];
	e_fp *zv=p->m2[6];
	e_fp *zu=p->m2[7];
	e_fp *zz=p->m2[8];
	e_fp t=0.0,s=0.0, ret=0.0;
	int n=p->N, loop=p->Loop;
	int j,k,l;

    /*
     *******************************************************************
     *   Kernel 18 - 2-D explicit hydrodynamics fragment
     *******************************************************************
     *       DO 75  L= 1,Loop
     *              T= 0.0037
     *              S= 0.0041
     *             KN= 6
     *             JN= n
     *       DO 70  k= 2,KN
     *       DO 70  j= 2,JN
     *        ZA(j,k)= (ZP(j-1,k+1)+ZQ(j-1,k+1)-ZP(j-1,k)-ZQ(j-1,k))
     *   .            *(ZR(j,k)+ZR(j-1,k))/(ZM(j-1,k)+ZM(j-1,k+1))
     *        ZB(j,k)= (ZP(j-1,k)+ZQ(j-1,k)-ZP(j,k)-ZQ(j,k))
     *   .            *(ZR(j,k)+ZR(j,k-1))/(ZM(j,k)+ZM(j-1,k))
     * 70    CONTINUE
     *       DO 72  k= 2,KN
     *       DO 72  j= 2,JN
     *        ZU(j,k)= ZU(j,k)+S*(ZA(j,k)*(ZZ(j,k)-ZZ(j+1,k))
     *   .                    -ZA(j-1,k) *(ZZ(j,k)-ZZ(j-1,k))
     *   .                    -ZB(j,k)   *(ZZ(j,k)-ZZ(j,k-1))
     *   .                    +ZB(j,k+1) *(ZZ(j,k)-ZZ(j,k+1)))
     *        ZV(j,k)= ZV(j,k)+S*(ZA(j,k)*(ZR(j,k)-ZR(j+1,k))
     *   .                    -ZA(j-1,k) *(ZR(j,k)-ZR(j-1,k))
     *   .                    -ZB(j,k)   *(ZR(j,k)-ZR(j,k-1))
     *   .                    +ZB(j,k+1) *(ZR(j,k)-ZR(j,k+1)))
     * 72    CONTINUE
     *       DO 75  k= 2,KN
     *       DO 75  j= 2,JN
     *        ZR(j,k)= ZR(j,k)+T*ZU(j,k)
     *        ZZ(j,k)= ZZ(j,k)+T*ZV(j,k)
     * 75    CONTINUE
     */
	 reinit_vec(p,zp,7*n);
	 reinit_vec(p,zq,7*n);
	 reinit_vec(p,zm,7*n);
	 reinit_vec(p,zr,7*n);
	 reinit_vec(p,zz,7*n);
	 zero_vec(zv,7*n);
	 zero_vec(zu,7*n);

    for ( l=1 ; l<=loop ; l++ ) {
        int kn = 6;
        int jn = n;
        t = 0.0000037;
        s = 0.0000041;
        for ( k=1 ; k<kn ; k++ ) {
          for ( j=1 ; j<jn ; j++ ) {
			int up=(k-1)*jn+j;
			int down=(k+1)*jn+j;
			int cur=k*jn+j;
              za[cur] = ( zp[down-1] +zq[down-1] -zp[cur-1] -zq[cur-1] )*
                         ( zr[cur] +zr[cur-1] ) / ( zm[cur-1] +zm[down-1]);
              zb[cur] = ( zp[cur-1] +zq[cur-1] -zp[cur] -zq[cur] ) *
                         ( zr[cur] +zr[up] ) / ( zm[cur] +zm[cur-1]);
			  if (!th_isfinite(za[cur])) za[cur]=FPCONST(1.0);
			  if (!th_isfinite(zb[cur])) zb[cur]=FPCONST(1.0);
          }
        }
        for ( k=1 ; k<kn ; k++ ) {
            for ( j=1 ; j<jn ; j++ ) {
			int up=(k-1)*jn+j;
			int down=(k+1)*jn+j;
			int cur=k*jn+j;
                zu[cur] += s*( za[cur]   *( zz[cur] - zz[cur+1] ) -
                                za[cur-1] *( zz[cur] - zz[cur-1] ) -
                                zb[cur]   *( zz[cur] - zz[up] ) +
                                zb[down] *( zz[cur] - zz[down] ) );
                zv[cur] += s*( za[cur]   *( zr[cur] - zr[cur+1] ) -
                                za[cur-1] *( zr[cur] - zr[cur-1] ) -
                                zb[cur]   *( zr[cur] - zr[up] ) +
                                zb[down] *( zr[cur] - zr[down] ) );
            }
        }
        for ( k=1 ; k<kn ; k++ ) {
            for ( j=1 ; j<jn ; j++ ) {
				int cur=k*jn+j;
                zr[cur] = zr[cur] + t*zu[cur];
                zz[cur] = zz[cur] + t*zv[cur];
			  if (!th_isfinite(zr[cur])) zr[cur]=FPCONST(1.0);
			  if (!th_isfinite(zz[cur])) zz[cur]=FPCONST(1.0);
            }
        }
		ret+=get_array_feedback(zr,kn*jn);
		ret+=get_array_feedback(zz,kn*jn);
    }
	return ret;
}

e_fp lin_recurrence(loops_params *p) {
	e_fp *sa=p->v[0],  *sb=p->v[1];
	e_fp *b5=p->v[2];
	e_fp stb5;
	int n=p->N, loop=p->Loop;
       /*
     *******************************************************************
     *   Kernel 19 -- general linear recurrence equations
     *******************************************************************
     *               KB5I= 0
     *           DO 194 L= 1,Loop
     *           DO 191 k= 1,n
     *         B5(k+KB5I)= SA(k) +STB5*SB(k)
     *               STB5= B5(k+KB5I) -STB5
     *191        CONTINUE
     *192        DO 193 i= 1,n
     *                  k= n-i+1
     *         B5(k+KB5I)= SA(k) +STB5*SB(k)
     *               STB5= B5(k+KB5I) -STB5
     *193        CONTINUE
     *194 CONTINUE
     */
	int kb5i = 0,l,i,k;
	reinit_vec(p,sa,n);
	reinit_vec(p,sb,n);
	stb5=sa[0]/sa[1];/* small random number */
    for ( l=1 ; l<=loop ; l++ ) {
		reinit_vec(p,sa,n);
        for ( k=0 ; k<n ; k++ ) {
            b5[k+kb5i] = sa[k] + stb5*sb[k];
            stb5 = b5[k+kb5i] - stb5;
        }
        for ( i=1 ; i<=n ; i++ ) {
            k = n - i ;
            b5[k+kb5i] = sa[k] + stb5*sb[k];
            stb5 = b5[k+kb5i] - stb5;
        }
#if (BMDEBUG && DEBUG_ACCURATE_BITS)
	th_printf("\nrec %d:",debug_counter++);
	th_print_fp(stb5);
#endif
   }
	return stb5;
}

e_fp ordinate_transport(loops_params *p) {
	e_fp *y=p->v[0],  *g=p->v[3];
	e_fp *u=p->v[1],  *v=p->v[4];
	e_fp *w=p->v[2],  *z=p->v[5];
	e_fp *xx=p->v[6];
	e_fp *vx=p->v[7];
	e_fp *x=p->v[8];
	e_fp di,dn,dk;
	e_fp	ret=0.0;
    /*
     *******************************************************************
     *   Kernel 20 -- Discrete ordinates transport, conditional recurrence on xx
     *******************************************************************
     *    DO 20 L= 1,Loop
     *    DO 20 k= 1,n
     *         DI= Y(k)-G(k)/( XX(k)+DK)
     *         DN= 0.2
     *         IF( DI.NE.0.0) DN= MAX( S,MIN( Z(k)/DI, T))
     *       X(k)= ((W(k)+V(k)*DN)* XX(k)+U(k))/(VX(k)+V(k)*DN)
     *    XX(k+1)= (X(k)- XX(k))*DN+ XX(k)
     * 20 CONTINUE
     */
	int n=p->N, loop=p->Loop;
	int l,k;
	e_fp t,s;
	reinit_vec(p,y,n);
	reinit_vec(p,g,n);
	reinit_ordinate_cvec(p,u,n);
	reinit_ordinate_cvec(p,vx,n);
	reinit_vec(p,v,n);
	reinit_ordinate_cvec(p,w,n);
	reinit_vec(p,z,n);
	reinit_vec(p,xx,3);
	t=xx[0];
	s=xx[1];
	dk=xx[2];
    for ( l=1 ; l<=loop ; l++ ) {
		reinit_vec(p,xx,1); /* start a new input each outer loop iteration otherwise only last iteration will be executed */
        for ( k=0 ; k<n-1 ; k++ ) {
            di = y[k] - g[k] / ( xx[k] + dk );
            dn = 0.2;
            if ( di ) {
                dn = z[k]/di ;
                if ( t < dn ) dn = t;
                if ( s > dn ) dn = s;
            }
            x[k] = ( ( w[k] + v[k]*dn )* xx[k] + u[k] ) / ( vx[k] + v[k]*dn );
            xx[k+1] = ( x[k] - xx[k] )* dn + xx[k];
        }
		ret+=xx[n-1]; /* return an output value dependent on each loop iteration */
#if (BMDEBUG && DEBUG_ACCURATE_BITS)
	th_printf("\nord %d:",debug_counter++);
	th_print_fp(ret);
#endif
    }
	return ret;
}

e_fp matmul(loops_params *p) {
	e_fp *px=p->m2[0];
	e_fp *vy=p->m2[1];
	e_fp *cx=p->m2[2];
	int n=p->N, loop=p->Loop;
	int i,j,k,l;
	e_fp ret;
	zero_vec(px,25*n);
	reinit_vec_limited(p,vy,25*n);
	reinit_vec_limited(p,cx,25*n+loop);
   /*
     *******************************************************************
     *   Kernel 21 -- matrix*matrix product
     *******************************************************************
     *    DO 21 L= 1,Loop
     *    DO 21 k= 1,25
     *    DO 21 i= 1,25
     *    DO 21 j= 1,n
     *    PX(i,j)= PX(i,j) +VY(i,k) * CX(k,j)
     * 21 CONTINUE
     */
	/*SG: Add dependency on outer loop to CX so compiler cannot optimize it away to a multiply on the final result. */
    for ( l=1 ; l<=loop ; l++ ) {
#if USE_RVV
      for (j=0; j<n; j++) {
        for (i=0; i<25; ) {
          size_t vl = __riscv_vsetvl_e32m8(25-i);
          vfloat32m8_t pxs = __riscv_vle32_v_f32m8(&px[j*25+i], vl);
          for (k=0; k<25; k++) {
            float cxs = cx[j*25+k+l];
            vfloat32m8_t pys = __riscv_vle32_v_f32m8(&vy[k*n+i], vl);
            pxs = __riscv_vfmacc_vf_f32m8(pxs, cxs, pys, vl);
          }
          __riscv_vse32_v_f32m8(&px[j*25+i], pxs, vl);
          i += vl;
        }
      }
#else
        for ( k=0 ; k<25 ; k++ ) {
            for ( i=0 ; i<25 ; i++ ) {
                for ( j=0 ; j<n ; j++ ) {
                    px[j*25+i] += vy[k*n+i] * cx[j*25+k+l];
                }
            }
        }
#endif
    }
	ret=get_array_feedback(px,n*25);
	return ret;
}

e_fp planckian(loops_params *p) {
	e_fp *y=p->v[0];
	e_fp *u=p->v[1],  *v=p->v[3];
	e_fp *w=p->v[2],  *x=p->v[4];
	e_fp	ret=FPCONST(0.0);
	int n=p->N, loop=p->Loop,l,k;
    e_fp expmax = FPCONST(20.0);
    /*
     *******************************************************************
     *   Kernel 22 -- Planckian distribution
     *******************************************************************
     *     EXPMAX= 20.0
     *       U(n)= 0.99*EXPMAX*V(n)
     *    DO 22 L= 1,Loop
     *    DO 22 k= 1,n
     *                                          Y(k)= U(k)/V(k)
     *       W(k)= X(k)/( EXP( Y(k)) -1.0)
     * 22 CONTINUE
     */

	reinit_vec(p,u,n);
	reinit_vec(p,x,n);
	reinit_vec(p,v,n);
    u[n-1] = FPCONST(0.99)*expmax*v[n-1];
    for ( l=1 ; l<=loop ; l++ ) {
		reinit_vec(p,v,n); /* must reinit otherwise compiler will only exec last loop */
        for ( k=0 ; k<n ; k++ ) {
            y[k] = u[k] / v[k];
            w[k] = x[k] / ( th_exp( y[k] ) - FPCONST(1.0) );
        }
		ret+=get_array_feedback(w,n); /* return a value that depends on each compute in the loop */
    }
	return ret;
}

e_fp hydro_2d_implicit(loops_params *p) {
	e_fp *za=p->m2[0];
	e_fp *zr=p->m2[1];
	e_fp *zu=p->m2[2];
	e_fp *zb=p->m2[3];
	e_fp *zv=p->m2[4];
	e_fp *zz=p->m2[5];
	int n=p->N, loop=p->Loop;
	int j,k,l;
	e_fp ret;
	zero_vec(za,6*n);
	reinit_vec(p,zr,6*n);
	reinit_vec(p,zb,6*n);
	reinit_vec(p,zz,6*n);
	reinit_vec(p,zv,6*n);
	reinit_vec(p,zu,6*n);
    /*
     *******************************************************************
     *   Kernel 23 -- 2-D implicit hydrodynamics fragment
     *******************************************************************
     *    DO 23  L= 1,Loop
     *    DO 23  j= 2,6
     *    DO 23  k= 2,n
     *          QA= ZA(k,j+1)*ZR(k,j) +ZA(k,j-1)*ZB(k,j) +
     *   .          ZA(k+1,j)*ZU(k,j) +ZA(k-1,j)*ZV(k,j) +ZZ(k,j)
     * 23  ZA(k,j)= ZA(k,j) +.175*(QA -ZA(k,j))
     */

    for ( l=1 ; l<=loop ; l++ ) {
        for ( j=1 ; j<6 ; j++ ) {
            for ( k=1 ; k<n ; k++ ) {
				int cur=j*n+k;
				int up=(j-1)*n+k;
				int down=(j+1)*n+k;
                e_fp qa = za[down]*zr[cur] + za[up]*zb[cur] +
                     za[cur+1]*zu[cur] + za[cur-1]*zv[cur] + zz[cur];
                za[cur] += FPCONST(0.175)*( qa - za[cur] );
            }
        }
    }
	ret=get_array_feedback(za,n); /* return a value that depends on each compute in the loop */
	return ret;
}

e_fp firstmin(loops_params *p) {
	e_fp *x=p->v[0];
	int	ret=0;
	int n=p->N, loop=p->Loop;
	int l,k,m;
    /*
     *******************************************************************
     *   Kernel 24 -- find location of first minimum in array
     *******************************************************************
     *     X( n/2)= -1.0E+10
     *    DO 24  L= 1,Loop
     *           m= 1
     *    DO 24  k= 2,n
     *          IF( X(k).LT.X(m))  m= k
     * 24 CONTINUE
     */
    for ( l=1 ; l<=loop ; l++ ) {
		reinit_vec(p,x,n); /* must reinit otherwise compiler can execute only the last loop */
		x[n/2] = FPCONST(-1.0e+10);
        m = 0;
        for ( k=1 ; k<n ; k++ ) {
            if ( x[k] < x[m] ) m = k;
        }
		ret+=m;
    }
	return (e_fp)ret;
}
