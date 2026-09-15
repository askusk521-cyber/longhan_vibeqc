"""Shell-shared lowering of the existing weighted DF derivative moment DAG.

The prototype covers the seven non-SSS s/p classes. Geometry and Boys values
come from the scalar generated evaluator; only ownership and reuse change.
No derivative tensor is produced: the consumer accumulates six independent
center coordinates and recovers the auxiliary center by translation.
"""

from itertools import product

from .cuda import CudaEmitter
from .df_derivatives import axis_polynomial

PROTOTYPE_CLASSES = tuple(a for a in product(range(2), repeat=3) if any(a))


def axis_cache_layout(angular):
    """Pack variable-length polynomials, leaving only unused doubly-raised slots.

    The rectangular indexing makes device lookup arithmetic independent of
    component tables. Only A or B is raised for a first derivative, so their
    jointly raised boundary is reserved but never generated or read.
    """
    offsets, size = {}, 0
    for powers in product(
        range(angular[0] + 2), range(angular[1] + 2), range(angular[2] + 1)
    ):
        offsets[powers] = size
        size += sum(powers) + 1
    return offsets, size


def emit_df_shell_derivatives_cuda():
    """Emit class-specialized cache construction from the shared polynomial IR."""
    lines = [
        r"""// Generated shell-shared weighted DF derivatives.
#ifndef VIBEQC_GENERATED_DF_SHELL_DERIVATIVES_CUH
#define VIBEQC_GENERATED_DF_SHELL_DERIVATIVES_CUH
#include "generated_df_derivatives.cuh"
namespace vibeqc::scf::generated_df_shell {
namespace scalar = generated_df_derivatives;
template<unsigned A,unsigned B,unsigned C> struct Shell;
struct Contracted { double gradient[3][3]; };

/** CCA index of one normalized expansion's Cartesian component. */
__device__ __forceinline__ unsigned cartesian_index(unsigned l,const unsigned char* a) {
  const unsigned r=l-a[0];return r*(r+1)/2+a[2];
}
template<unsigned L>
__device__ __forceinline__ scalar::Angular angular(unsigned i) {
  for(unsigned r=0;r<=L;++r)
    if(i<(r+1)*(r+2)/2) {
      const unsigned z=i-r*(r+1)/2;
      return {L-r,r-z,z};
    }
  return {0,0,0};
}

template<unsigned A,unsigned B,unsigned C>
struct Moments {
  static constexpr unsigned na=(A+1)*(A+2)/2,nb=(B+1)*(B+2)/2,nc=(C+1)*(C+2)/2;
  static constexpr unsigned components=na*nb*nc;
  static constexpr unsigned rows=B+2,columns=C+1;
  static constexpr unsigned axis_size=(A+2)*rows*columns*(A+B+C+4)/2;
  /** Translation recovers the auxiliary center from the two independent ones. */
  __device__ static Contracted finish(const double* independent) {
    Contracted result{};
    for(unsigned axis=0;axis<3;++axis) {
      result.gradient[0][axis]=independent[axis];
      result.gradient[1][axis]=independent[3+axis];
      result.gradient[2][axis]=-independent[axis]-independent[3+axis];
    }
    return result;
  }
  /** Sum the exact coefficient lengths of preceding rectangular entries. */
  __device__ __forceinline__ static unsigned offset(unsigned a,unsigned b,unsigned c) {
    return a*rows*columns*(a+rows+columns-1)/2
         + b*columns*(2*a+b+columns)/2 + c*(a+b+1)+c*(c-1)/2;
  }
  /** Fuse the differentiated axis before integrating shared other-axis moments.
   * alpha/beta and all external weights stay fixed under nuclear response.
   */
  __device__ __forceinline__ static void accumulate(unsigned item,double alpha,double beta,
      const scalar::Geometry& g,const double* cache,double weight,double* out) {
    const auto a=angular<A>(item/nb/nc),b=angular<B>(item/nc%nb),c=angular<C>(item%nc);
    unsigned degree[3],index[3];
    for(unsigned axis=0;axis<3;++axis) {
      const auto x=scalar::power(a,axis),y=scalar::power(b,axis),z=scalar::power(c,axis);
      degree[axis]=x+y+z;index[axis]=axis*axis_size+offset(x,y,z);
    }
    for(unsigned axis=0;axis<3;++axis) {
      const unsigned other=(axis+1)%3,last=(axis+2)%3;
      const auto x=scalar::power(a,axis),y=scalar::power(b,axis),z=scalar::power(c,axis);
      const double* v=cache+index[other];const double* w=cache+index[last];
      for(unsigned center=0;center<2;++center) {
        const unsigned n=center==0?x:y;
        const double exponent=center==0?alpha:beta;
        const double* raised=cache+axis*axis_size+offset(x+(center==0),y+(center==1),z);
        const double* lowered=n?cache+axis*axis_size+offset(x-(center==0),y-(center==1),z):raised;
        double value=0;
        for(unsigned i=0;i<=degree[axis]+1;++i) {
          const double derivative=2*exponent*raised[i]-(n && i<degree[axis]?n*lowered[i]:0.0);
          for(unsigned j=0;j<=degree[other];++j)
            for(unsigned k=0;k<=degree[last];++k)
              value+=derivative*v[j]*w[k]*g.f[i+j+k];
        }
        out[center*3+axis]+=weight*g.prefactor*value;
      }
    }
  }
};
"""
    ]
    for angular in PROTOTYPE_CLASSES:
        offsets, _ = axis_cache_layout(angular)
        parameters = ",".join(map(str, angular))
        lines += [
            f"template<> struct Shell<{parameters}> : Moments<{parameters}> {{",
            "  __device__ static void prepare_axis(const scalar::Geometry& g,unsigned axis,double* out) {",
            "    const double pa=g.pa[axis],pb=g.pb[axis],dx=g.dx[axis];",
            "    const double sx=g.sx,sy=g.sy,ip=g.ip,iq=g.iq;",
        ]
        for powers, offset in offsets.items():
            if powers[0] == angular[0] + 1 and powers[1] == angular[1] + 1:
                continue
            graph, roots = axis_polynomial(*powers)
            emitter = CudaEmitter(graph, {})
            emitter.emit(roots)
            lines += ["    {", *emitter.lines]
            lines += [
                f"      out[{offset + i}]={emitter.reference(root)};"
                for i, root in enumerate(roots)
            ]
            lines += ["    }"]
        lines += ["  }", "};"]
    lines += ["template<class Function> void for_each_class(Function function) {"]
    for angular in PROTOTYPE_CLASSES:
        lines += [f"  function.template operator()<{','.join(map(str, angular))}>();"]
    lines += ["}", "} // namespace vibeqc::scf::generated_df_shell", "#endif", ""]
    return "\n".join(lines)
