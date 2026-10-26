"""Campos cuadráticos físicos y soluciones lineales nativas, sin reparametrizarlas.

Convención: P''+(w*w/4-a)P=0. L: tL''+L'/2+(t-a)L=0.
Las derivadas en el corte o en extremos singulares se dejan como NaN.
"""
from functools import lru_cache
from pathlib import Path
import json
import shutil
import subprocess
import time
import tempfile
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

PARITIES = [('even','even','par–par'),('even','odd','par–impar'),
            ('odd','even','impar–par'),('odd','odd','impar–impar')]
K_T = 0.5
SPAN = 25.0
LINEAR_NEGATIVE_POLICY = 'empty'

def save_figure_atomic(fig,path,**kwargs):
    """Finaliza el archivo en disco temporal antes de sustituir la salida existente."""
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    with tempfile.TemporaryDirectory(prefix='weber-export-') as folder:
        stage=Path(folder)/path.name
        fig.savefig(stage,**kwargs)
        pending=path.with_name('.'+path.name+'.partial')
        shutil.copyfile(stage,pending)
        pending.replace(path)
    return path

@lru_cache(maxsize=16)
def quadratic_basis(a, limit):
    def ode(w,z):
        c=a-w*w/4
        return [z[1],c*z[0],z[3],c*z[2]]
    sol=solve_ivp(ode,(0.,limit),[1.,0.,0.,1.],method='DOP853',
                  rtol=2e-12,atol=2e-14,dense_output=True,max_step=.12)
    if not sol.success: raise RuntimeError(sol.message)
    return sol

def P(w,a,parity='even',derivative=False):
    if parity not in ('even','odd'): raise ValueError('Paridad inválida')
    w=np.asarray(w,float)
    limit=max(1.,float(np.ceil(np.max(np.abs(w))+.1)))
    index=(0 if parity=='even' else 2)+int(derivative)
    z=quadratic_basis(float(a),limit).sol(np.abs(w).ravel())[index].reshape(w.shape)
    if (parity=='odd') != bool(derivative): z=z*np.sign(w)
    return z

def inverse_quadratic(x,y,side=1):
    x,y=np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float))
    r=np.hypot(x,y)
    xi=np.array(np.sqrt(np.maximum(r+x,0.)),copy=True)
    eta=np.array(np.where(y==0,side,np.sign(y))*np.sqrt(np.maximum(r-x,0.)),copy=True)
    np.divide(y,xi,out=eta,where=(x>=0)&(xi>0))
    np.divide(np.abs(y),np.abs(eta),out=xi,where=(x<0)&(eta!=0))
    return xi,eta

def field_quadratic(x,y,a,p,q,side=1,quantity='field',k_t=K_T):
    xi,eta=inverse_quadratic(x,y,side)
    root=np.sqrt(2*k_t)
    f,g=P(root*xi,a,p),P(root*eta,-a,q)
    if quantity=='field': return f*g
    fp,gp=root*P(root*xi,a,p,True),root*P(root*eta,-a,q,True)
    den=xi*xi+eta*eta
    numerator=fp*xi*g-f*gp*eta if quantity=='d1' else fp*eta*g+f*gp*xi
    if quantity not in ('d1','d2'): raise ValueError('quantity: field, d1 o d2')
    result=np.full(np.shape(den),np.nan)
    np.divide(numerator,den,out=result,where=den>0)
    # Para paridades mixtas no hay una derivada clásica sobre el corte.
    if p!=q:
        result=np.where((np.asarray(y)==0)&(np.asarray(x)<0),np.nan,result)
    return result

def _frobenius_coeffs(a,alpha,order=12):
    b=2*alpha+.5
    c=[2. if alpha else 1.]
    for n in range(order):
        c.append((a*c[n]-(c[n-1] if n else 0.))/((n+1)*(n+b)))
    return np.asarray(c)

@lru_cache(maxsize=16)
def linear_basis(a,limit):
    """Integra directamente la EDO lineal regularizada G, nunca llama P."""
    eps=1e-5
    coeff=[_frobenius_coeffs(a,alpha) for alpha in (0.,.5)]
    def initial(c):
        return [np.polynomial.polynomial.polyval(eps,c),
                np.polynomial.polynomial.polyval(eps,np.arange(1,len(c))*c[1:])]
    z0=initial(coeff[0])+initial(coeff[1])
    def ode(t,z):
        return [z[1],((a-t)*z[0]-.5*z[1])/t,
                z[3],((a-t)*z[2]-1.5*z[3])/t]
    sol=solve_ivp(ode,(eps,max(limit,1.)),z0,method='DOP853',
                  rtol=2e-12,atol=2e-14,dense_output=True,max_step=.12)
    if not sol.success: raise RuntimeError(sol.message)
    return sol,coeff,eps

def linear_factor(t,a,parity='even',derivative=False):
    """L(t), dL/dt para t>=0; la parte negativa queda vacía.

    Frobenius alfa=0 ('even') o alfa=1/2 ('odd'). No es paridad t -> -t.
    """
    if parity not in ('even','odd'): raise ValueError('Paridad inválida')
    t=np.asarray(t,float)
    good=t>=0
    safe=np.where(good,t,0.)
    alpha=0. if parity=='even' else .5
    index=0 if parity=='even' else 2
    sol,coeff,eps=linear_basis(float(a),max(1.,float(np.ceil(np.max(safe)+.1))))
    states=sol.sol(np.maximum(safe,eps).ravel()).reshape((4,)+t.shape)
    g,gp=states[index],states[index+1]
    c=coeff[index//2]
    near=safe<eps
    g=np.where(near,np.polynomial.polynomial.polyval(safe,c),g)
    gp=np.where(near,np.polynomial.polynomial.polyval(safe,np.arange(1,len(c))*c[1:]),gp)
    if derivative:
        if alpha:
            with np.errstate(divide='ignore',invalid='ignore'):
                result=.5*g/np.sqrt(safe)+np.sqrt(safe)*gp
            result=np.where(safe==0,np.nan,result)
        else: result=gp
    else: result=safe**alpha*g
    return np.where(good,result,np.nan)

def field_linear(t,s,a,p,q,side=1,quantity='field'):
    """Producto nativo L_p(t;a)L_q(s;-a). t,s son los ejes, sin mapa cartesiano."""
    if quantity not in ('field','d1','d2'): raise ValueError('quantity inválida')
    f=linear_factor(t,a,p,quantity=='d1')
    g=linear_factor(s,-a,q,quantity=='d2')
    return f*g

def evaluate(system,x,y,a,p,q,quantity='field',side=1):
    if system in ('quadratic','cuadraticas'):
        return field_quadratic(x,y,a,p,q,side,quantity)
    if system in ('linear','lineales'):
        return field_linear(x,y,a,p,q,side,quantity)
    raise ValueError('system debe ser quadratic o linear')

def axis_labels(system,quantity='field'):
    linear=system in ('linear','lineales')
    xy=('t','s') if linear else ('x','y')
    z={'field':'campo','d1':f'dψ/d{xy[0]}','d2':f'dψ/d{xy[1]}'}[quantity]
    return *xy,z

def pieces(system,span=SPAN,n=151):
    """Piezas desconectadas: no hay triángulos que atraviesen cortes."""
    if system in ('linear','lineales'):
        axis=np.linspace(0.,span,n)
        x,y=np.meshgrid(axis,axis)
        return [(x,y,1)]
    axis=np.linspace(-span,span,n)
    half=np.linspace(0.,span,(n+1)//2)
    x,y=np.meshgrid(axis,half)
    return [(x,y,1),(x,-y,-1)]

def normalized_pieces(system,a,quantity='field',span=SPAN,n=151):
    data=[]
    for p,q,name in PARITIES:
        group=[]
        for x,y,side in pieces(system,span,n):
            z=evaluate(system,x,y,a,p,q,quantity,side)
            group.append((x,y,z,side))
        finite=[np.abs(z[np.isfinite(z)]) for _,_,z,_ in group]
        scale=max((float(np.max(v)) for v in finite if v.size),default=1.) or 1.
        data.append((name,scale,[(x,y,z/scale,side) for x,y,z,side in group]))
    return data

def compact_figure(system,a,quantity='field',figsize=(12,8),dpi=120,span=SPAN):
    # Los ejes 3D se colocan manualmente para evitar los márgenes de constrained_layout.
    fig=plt.figure(figsize=figsize,dpi=dpi,facecolor='white')
    axes=[]
    for i,(_,_,name) in enumerate(PARITIES):
        x0=.06+(i%2)*.40; y0=.035+(1-i//2)*.445
        ax=fig.add_axes([x0,y0,.48,.445],projection='3d')
        ax.set_proj_type('ortho')
        ax.view_init(29,-58)
        ax.set_box_aspect((1.25,1.,.58),zoom=1.1)
        ax.set(xlim=(-span,span),ylim=(-span,span),zlim=(-1.05,1.05),title=name)
        xl,yl,zl=axis_labels(system,quantity)
        ax.set_xlabel(xl,labelpad=-1);ax.set_ylabel(yl,labelpad=-1)
        ax.set_zlabel(zl+' / C',labelpad=0)
        ax.set_xticks([-.8*span,0,.8*span]);ax.set_yticks([-.8*span,0,.8*span])
        ax.set_zticks([-1,-.5,0,.5,1])
        ax.tick_params(labelsize=7,pad=-1)
        axes.append(ax)
    title=('Cuadráticas · plano físico' if system in ('quadratic','cuadraticas')
           else 'Lineales · variables nativas (negativos vacíos)')
    fig.suptitle(f'{title} · {axis_labels(system,quantity)[2]} · a={a:g}',y=.983,fontsize=14)
    return fig,axes

def draw_surfaces(axes,data,n=151):
    artists=[]
    for ax,(name,scale,group) in zip(axes,data):
        for x,y,z,side in group:
            artist=ax.plot_surface(x,y,z,color='#2783ad' if side==1 else '#df9441',
                                   linewidth=0,antialiased=True,shade=True,
                                   rcount=z.shape[0],ccount=z.shape[1])
            artists.append(artist)
        ax.set_title(f'{name} · C={scale:.3g}',fontsize=10,pad=0)
    return artists

def save_atlas(system,a,quantity='field',output_directory='Weber_3D_resultados',
               n=151,span=SPAN,dpi=220,show=True):
    output_directory=Path(output_directory);output_directory.mkdir(parents=True,exist_ok=True)
    fig,axes=compact_figure(system,a,quantity,span=span)
    draw_surfaces(axes,normalized_pieces(system,a,quantity,span,n))
    path=output_directory/f'{system}_{quantity}_a_{a:+g}.png'
    fig.savefig(path,dpi=dpi,facecolor='white')
    if show: plt.show()
    plt.close(fig)
    return path

def animate_weber_3d(system='quadratic',a_initial=-4.,a_final=4.,duration_seconds=8,
                     fps=60,spatial_domain=(-25.,25.),pts_phys=101,
                     output=None,width=3840,height=2160,quantity='field',crf=14):
    """Fotogramas reales a cada a, sin interpolación temporal ni escalado posterior."""
    if width%2 or height%2: raise ValueError('Dimensiones pares requeridas')
    if spatial_domain[0] != -spatial_domain[1]: raise ValueError('Dominio simétrico requerido')
    if shutil.which('ffmpeg') is None: raise RuntimeError('Falta ffmpeg')
    span=float(spatial_domain[1]); output=Path(output or f'Weber_{system}_3D_4K_60fps.mp4')
    output.parent.mkdir(parents=True,exist_ok=True)
    total=int(round(duration_seconds*fps))
    fig,axes=compact_figure(system,a_initial,quantity,figsize=(width/160,height/160),dpi=160,span=span)
    # Escalar tipografía con los píxeles de salida manteniendo la composición del atlas.
    ratio=width/1920
    for ax in axes:
        ax.tick_params(labelsize=7*ratio,pad=2*ratio)
        for lab in (ax.xaxis.label,ax.yaxis.label,ax.zaxis.label): lab.set_fontsize(10*ratio)
        ax.xaxis.labelpad=10*ratio;ax.yaxis.labelpad=10*ratio
        ax.zaxis.labelpad=17*ratio
    fig._suptitle.set_fontsize(16*ratio)
    writer=FFMpegWriter(fps=fps,codec='libx264',bitrate=-1,
                       extra_args=['-preset','fast','-crf',str(crf),'-pix_fmt','yuv420p',
                                   '-movflags','+faststart','-threads','4'])
    started=time.monotonic()
    artists=[]
    with writer.saving(fig,str(output),dpi=160):
        for frame,a in enumerate(np.linspace(a_initial,a_final,total)):
            for artist in artists: artist.remove()
            data=normalized_pieces(system,float(a),quantity,span,pts_phys)
            artists=draw_surfaces(axes,data)
            for ax in axes: ax.title.set_fontsize(11*ratio)
            title=('Cuadráticas · x,y' if system=='quadratic' else 'Lineales nativas · t,s ≥ 0')
            fig._suptitle.set_text(f'{title} · [-{span:g},{span:g}] · a={a:+.3f} · {axis_labels(system,quantity)[2]} / C')
            writer.grab_frame(facecolor='white')
            if frame%10==0 or frame==total-1:
                print(f'{system}: {frame+1}/{total}; {time.monotonic()-started:.1f}s',flush=True)
    plt.close(fig)
    manifest={'system':system,'a_initial':a_initial,'a_final':a_final,
              'width':width,'height':height,'fps':fps,'frames':total,
              'duration_seconds':duration_seconds,'grid_points':pts_phys,
              'quantity':quantity,'domain':list(spatial_domain),
              'normalization':'por panel y fotograma','linear_negative_policy':'empty',
              'elapsed_seconds':time.monotonic()-started}
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2))
    return str(output)

def animate_weber_evolution_fullhd(system='quadratic',a_initial=-4,a_final=4,
        duration_seconds=8,fps=60,spatial_domain=(-25,25),pts_phys=700,output=None,
        width=3840,height=2160):
    """Animación 2D original reparada, ahora con salida 4K y lineales nativas."""
    output=Path(output or f'Weber_{system}_2D_4K_60fps.mp4')
    output.parent.mkdir(exist_ok=True,parents=True)
    fig,axes,images=heatmap_figure(system,spatial_domain,width,height)
    grid=prepare_heatmap_grid(system,spatial_domain,pts_phys)
    writer=FFMpegWriter(fps=fps,codec='libx264',bitrate=-1,
        extra_args=['-preset','fast','-crf','14','-pix_fmt','yuv420p','-movflags','+faststart','-threads','4'])
    total=int(round(duration_seconds*fps));started=time.monotonic()
    with writer.saving(fig,str(output),dpi=160):
        for frame,a in enumerate(np.linspace(a_initial,a_final,total)):
            fields=heatmap_fields(system,float(a),grid)
            for ax,image,field,(_,_,name) in zip(axes,images,fields,PARITIES):
                scale=np.nanmax(abs(field)) or 1.
                image.set_data(field/scale);ax.set_title(f'{name} · C={scale:.3g}',fontsize=18)
            fig._suptitle.set_text(f'Weber {system} · a={a:+.3f} · amplitud / C')
            writer.grab_frame(facecolor='white')
            if frame%20==0 or frame==total-1:
                print(f'{system} 2D: {frame+1}/{total}; {time.monotonic()-started:.1f}s',flush=True)
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(dict(system=system,kind='2D',width=width,
        height=height,fps=fps,frames=total,duration_seconds=duration_seconds,
        a_initial=a_initial,a_final=a_final,grid_points=pts_phys,domain=list(spatial_domain)),indent=2))
    return str(output)

def prepare_heatmap_grid(system,span,points):
    axis=np.linspace(*span,points)
    x,y=np.meshgrid(axis,axis)
    if system=='quadratic':
        xi,eta=inverse_quadratic(x,y)
        return np.sqrt(2*K_T)*xi,np.sqrt(2*K_T)*eta
    return x,y

def heatmap_fields(system,a,grid):
    u,v=grid
    if system=='quadratic':
        limit=max(1.,float(np.ceil(max(np.max(u),np.max(abs(v)))+.1)))
        axis=np.linspace(0,limit,16000)
        values1=quadratic_basis(a,limit).sol(axis)
        values2=quadratic_basis(-a,limit).sol(axis)
        f={p:CubicSpline(axis,values1[index])(u) for p,index in [('even',0),('odd',2)]}
        g={p:CubicSpline(axis,values2[index])(abs(v))*(np.sign(v) if p=='odd' else 1.)
           for p,index in [('even',0),('odd',2)]}
    else:
        # Valores nativos calculados sólo en las dos mallas 1D, sin mapa de coordenadas.
        f={p:linear_factor(u[0],a,p)[None,:] for p in ('even','odd')}
        g={p:linear_factor(v[:,0],-a,p)[:,None] for p in ('even','odd')}
    return [f[p]*g[q] for p,q,_ in PARITIES]

def heatmap_figure(system,span,width=3840,height=2160):
    fig=plt.figure(figsize=(width/160,height/160),dpi=160,facecolor='white')
    axes=[];images=[]
    # Un bloque cuadrado central, con sólo un pequeño pasillo entre los cuatro mapas.
    ratio=width/height
    h=.405;w=h/ratio
    left=(1-2*w-.035)/2
    for i,(_,_,name) in enumerate(PARITIES):
        ax=fig.add_axes([left+(i%2)*(w+.035),.055+(1-i//2)*.45,w,h])
        image=ax.imshow(np.zeros((2,2)),origin='lower',extent=(*span,*span),
            cmap='RdBu_r',vmin=-1,vmax=1,interpolation='lanczos',aspect='equal')
        xl,yl,_=axis_labels(system)
        ax.set(xlabel=xl if i>=2 else '',ylabel=yl if i%2==0 else '',
               title=name,xticks=[-25,0,25],yticks=[-25,0,25])
        ax.tick_params(labelsize=13);ax.xaxis.label.set_size(15);ax.yaxis.label.set_size(15)
        axes.append(ax);images.append(image)
    fig.suptitle(f'Weber {system}',fontsize=24,y=.985)
    cax=fig.add_axes([left+2*w+.085,.15,.012,.65])
    fig.colorbar(images[-1],cax=cax,label='amplitud / C')
    return fig,axes,images

def save_weber_snapshot_maxquality(a,system='quadratic',spatial_domain=(-25,25),
        pts_phys=1400,lookup_pts=9000,output_directory='Weber_Poster_Frames',dpi=600):
    """Captura 2D original reparada; no depende de definiciones eliminadas."""
    folder=Path(output_directory);folder.mkdir(exist_ok=True,parents=True)
    span=float(spatial_domain[1])
    fig,axes,images=heatmap_figure(system,spatial_domain,width=3840,height=2160)
    fig.set_size_inches(16,9)
    fields=heatmap_fields(system,a,prepare_heatmap_grid(system,spatial_domain,pts_phys))
    for ax,image,field,(_,_,name) in zip(axes,images,fields,PARITIES):
        scale=np.nanmax(abs(field)) or 1.
        image.set_data(field/scale);ax.set_title(f'{name} · C={scale:.3g}',fontsize=18)
    fig._suptitle.set_text(f'Weber {system} · a={a:g}')
    paths=[]
    for ext in ('png','pdf','svg'):
        path=folder/f'Weber_{system}_a_{a:+g}.{ext}'
        # Rasterizar las superficies evita SVG/PDF con cientos de miles de vectores.
        if ext!='png':
            for ax in axes:
                for col in ax.collections: col.set_rasterized(True)
        # 3840x2160 a 160 dpi; al exportar a dpi=600 se obtiene mayor resolución del cartel.
        save_figure_atomic(fig,path,dpi=dpi,facecolor='white');paths.append(str(path))
    plt.close(fig)
    return tuple(paths)
