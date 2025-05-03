import numpy as np
import torch
import torch.nn.functional as F


def FDM_Darcy(u, a, D=1):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    dx = D / (size - 1)
    dy = dx

    # ux: (batch, size-2, size-2)
    ux = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
    uy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dy)

    # ax = (a[:, 2:, 1:-1] - a[:, :-2, 1:-1]) / (2 * dx)
    # ay = (a[:, 1:-1, 2:] - a[:, 1:-1, :-2]) / (2 * dy)
    # uxx = (u[:, 2:, 1:-1] -2*u[:,1:-1,1:-1] +u[:, :-2, 1:-1]) / (dx**2)
    # uyy = (u[:, 1:-1, 2:] -2*u[:,1:-1,1:-1] +u[:, 1:-1, :-2]) / (dy**2)

    a = a[:, 1:-1, 1:-1]
    # u = u[:, 1:-1, 1:-1]
    # Du = -(ax*ux + ay*uy + a*uxx + a*uyy)

    # inner1 = torch.mean(a*(ux**2 + uy**2), dim=[1,2])
    # inner2 = torch.mean(f*u, dim=[1,2])
    # return 0.5*inner1 - inner2

    aux = a * ux
    auy = a * uy
    auxx = (aux[:, 2:, 1:-1] - aux[:, :-2, 1:-1]) / (2 * dx)
    auyy = (auy[:, 1:-1, 2:] - auy[:, 1:-1, :-2]) / (2 * dy)
    Du = - (auxx + auyy)
    return Du


def darcy_loss(u, a):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    lploss = LpLoss(size_average=True)

    # index_x = torch.cat([torch.tensor(range(0, size)), (size - 1) * torch.ones(size), torch.tensor(range(size-1, 1, -1)),
    #                      torch.zeros(size)], dim=0).long()
    # index_y = torch.cat([(size - 1) * torch.ones(size), torch.tensor(range(size-1, 1, -1)), torch.zeros(size),
    #                      torch.tensor(range(0, size))], dim=0).long()

    # boundary_u = u[:, index_x, index_y]
    # truth_u = torch.zeros(boundary_u.shape, device=u.device)
    # loss_u = lploss.abs(boundary_u, truth_u)

    Du = FDM_Darcy(u, a)
    f = torch.ones(Du.shape, device=u.device)
    loss_f = lploss.rel(Du, f)

    # im = (Du-f)[0].detach().cpu().numpy()
    # plt.imshow(im)
    # plt.show()

    # loss_f = FDM_Darcy(u, a)
    # loss_f = torch.mean(loss_f)
    return loss_f

def FDM_burgers(u, v, D=1):
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    u = u.reshape(batchsize, nt, nx)
    dt = D / (nt-1)
    dx = D / (nx)

    # ux: (batch, size-2, size-2)
    ux = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
    ut = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dt)
    uxx = (ux[:, 2:, 1:-1] - ux[:, :-2, 1:-1]) / (2 * dx)
    ux=ux[:, 1:-1, 1:-1]
    u=u[:, 2:-2, 2:-2]
    ut=ut[:, 2:-2, 2:-2]

    Du = ut + (ux*u - v*uxx)[:,1:-1,:]
    return Du,ux, uxx, ut

def burgers_loss(u, u0,v, index_ic=None, p=None, q=None):
    batchsize = u.size(0)
    # lploss = LpLoss(size_average=True)

    Du, ux, uxx, ut = FDM_burgers(u,v)

    if index_ic is None:
        # u in on a uniform grid
        nt = u.size(1)
        nx = u.size(2)
        u = u.reshape(batchsize, nt, nx)

        index_t = torch.zeros(nx,).long()
        index_x = torch.tensor(range(nx)).long()
        boundary_u = u[:, index_t, index_x]

        # loss_bc0 = F.mse_loss(u[:, :, 0], u[:, :, -1])
        # loss_bc1 = F.mse_loss(ux[:, :, 0], ux[:, :, -1])
    else:
        # u is randomly sampled, 0:p are BC, p:2p are ic, 2p:2p+q are interior
        boundary_u = u[:, :p]
        batch_index = torch.tensor(range(batchsize)).reshape(batchsize, 1).repeat(1, p)
        u0 = u0[batch_index, index_ic]

        # loss_bc0 = F.mse_loss(u[:, p:p+p//2], u[:, p+p//2:2*p])
        # loss_bc1 = F.mse_loss(ux[:, p:p+p//2], ux[:, p+p//2:2*p])

    loss_ic = F.mse_loss(boundary_u, u0)
    f = torch.zeros(Du.shape, device=u.device)
    loss_f = F.mse_loss(Du, f)
    return loss_ic, loss_f

def FDM_Advec(s, ux, D=1):
    batchsize = s.size(0)
    nt = s.size(1)
    nx = s.size(2)

    s = s.reshape(batchsize, nt, nx)
    dt = D / (nt-1)
    dx = D / (nx-1)

    st = (s[:, :, 2:] - s[:, :, :-2]) / (2 * dx)
    sx = (s[:, 2:, :] - s[:, :-2, :]) / (2 * dt)
    st = st[:, 1:-1, :]
    ux = ux.reshape(batchsize, nx,1).repeat([1, 1,nt])
    ux = ux[:,1:-1,:]
    Du = st + (ux * sx)[:,:,1:-1]
    return Du

def Advec_loss(u, ux):
    batchsize = u.size(0)
    nx = u.size(1)
    nt = u.size(2)

    u = u.reshape(batchsize, nx, nt)
    x = torch.tensor(np.linspace(0, 1, nx), dtype=torch.float)
    x = x.reshape(1, nx).repeat([batchsize, 1])
    t = torch.tensor(np.linspace(0, 1, nt), dtype=torch.float)
    t = t.reshape(1, nt).repeat([batchsize, 1])
    f = lambda x: np.sin(np.pi * x)
    g = lambda t: np.sin(np.pi * t/2)

    # lploss = LpLoss(size_average=True)
    boundary_u1 = f(x).cuda()
    boundary_u2 = g(t).cuda()

    loss_u = F.mse_loss(boundary_u1, u[:,:,0])+F.mse_loss(boundary_u2, u[:,0,:])

    Du = FDM_Advec(u,ux)

    f = torch.zeros(Du.shape, device=u.device)
    loss_f = F.mse_loss(Du, f)
    return loss_u,loss_f


def FDM_NS_vorticity(w, v=1/40, t_interval=1.0):
    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)
    nt = w.size(3)
    device = w.device
    w = w.reshape(batchsize, nx, ny, nt)

    w_h = torch.fft.fft2(w, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx//2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1, N).reshape(1,N,N,1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N, 1).reshape(1,N,N,1)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap
    
    ux_h = 1j * k_y * f_h
    uy_h = -1j * k_x * f_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    u = torch.fft.irfft2(f_h[:, :, :k_max + 1], dim=[1, 2])
    ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
    uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
    wx = torch.fft.irfft2(wx_h[:, :, :k_max+1], dim=[1,2])
    wy = torch.fft.irfft2(wy_h[:, :, :k_max+1], dim=[1,2])
    wlap = torch.fft.irfft2(wlap_h[:, :, :k_max+1], dim=[1,2])
    # dx = 1.0/nx
    # dy = 1.0/ny
    # ux = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
    # uy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dy)
    # wx = (w[:, 2:, 1:-1] - w[:, :-2, 1:-1]) / (2 * dx)
    # wy = (w[:, 1:-1, 2:] - w[:, 1:-1, :-2]) / (2 * dy)
    dt = t_interval
    wt = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)
    # wt = wt[:,1:-1,1:-1]
    # wlap = wlap[:,1:-1,1:-1]
    # u = u[:,1:-1,1:-1]
    Du1 = wt + (ux*wx + ux*wy - v*wlap)[...,1:-1] #- forcing
    return Du1


def Autograd_Burgers(u, grid, v=1/100):
    from torch.autograd import grad
    gridt, gridx = grid

    ut = grad(u.sum(), gridt, create_graph=True)[0]
    ux = grad(u.sum(), gridx, create_graph=True)[0]
    uxx = grad(ux.sum(), gridx, create_graph=True)[0]
    Du = ut + ux*u - v*uxx
    return Du, ux, uxx, ut


def AD_loss(u, u0, grid,v, index_ic=None, p=None, q=None):
    batchsize = u.size(0)
    # lploss = LpLoss(size_average=True)

    Du, ux, uxx, ut = Autograd_Burgers(u, grid,v)

    if index_ic is None:
        # u in on a uniform grid
        nt = u.size(1)
        nx = u.size(2)
        u = u.reshape(batchsize, nt, nx)

        index_t = torch.zeros(nx,).long()
        index_x = torch.tensor(range(nx)).long()
        boundary_u = u[:, index_t, index_x]

        # loss_bc0 = F.mse_loss(u[:, :, 0], u[:, :, -1])
        # loss_bc1 = F.mse_loss(ux[:, :, 0], ux[:, :, -1])
    else:
        # u is randomly sampled, 0:p are BC, p:2p are ic, 2p:2p+q are interior
        boundary_u = u[:, :p]
        batch_index = torch.tensor(range(batchsize)).reshape(batchsize, 1).repeat(1, p)
        u0 = u0[batch_index, index_ic]

        # loss_bc0 = F.mse_loss(u[:, p:p+p//2], u[:, p+p//2:2*p])
        # loss_bc1 = F.mse_loss(ux[:, p:p+p//2], ux[:, p+p//2:2*p])

    loss_ic = F.mse_loss(boundary_u, u0)
    f = torch.zeros(Du.shape, device=u.device)
    loss_f = F.mse_loss(Du, f)
    return loss_ic, loss_f


class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def FDM_Burgers(u, v, D=1):
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    u = u.reshape(batchsize, nt, nx)
    dt = D / (nt-1)

    dx = D / (nx - 1)

    # ux: (batch, size-2, size-2)
    ux = (u[:, :, 2:] - u[:, :, :-2]) / (2 * dx)
    uxx =(ux[:,:, 2:] - ux[:, :, :-2]) / (2 * dx)
    ux=ux[:, :, 1:-1]
    ut = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dt)
    ut=ut[:, :, 2:-2]
    u = u[:, :, 2:-2]

    # u_h = torch.fft.fft(u, dim=2)
    # # Wavenumbers in y-direction
    # k_max = nx//2
    # k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=u.device),
    #                  torch.arange(start=-k_max, end=0, step=1, device=u.device)), 0).reshape(1,1,nx)
    # ux_h = 2j *np.pi*k_x*u_h
    # uxx_h = 2j *np.pi*k_x*ux_h
    # ux = torch.fft.irfft(ux_h[:, :, :k_max+1], dim=2, n=nx)
    # uxx = torch.fft.irfft(uxx_h[:, :, :k_max+1], dim=2, n=nx)
    # ut = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dt)
    Du = ut + (ux*u - v*uxx)[:,1:-1,:]
    return Du

def FDM_Burgers1(u, v, D=1):
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    u = u.reshape(batchsize, nt, nx)
    dt = D / (nt-1)

    dx = D / (nx - 1)

    # ux: (batch, size-2, size-2)
    ux = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dx)
    uxx =(ux[:,2:, :] - ux[:, :-2, :]) / (2 * dx)
    ux=ux[:, 1:-1, :]
    ut = (u[:, :, 2:] - u[:, :, :-2]) / (2 * dt)
    ut=ut[:, 2:-2, :]
    u = u[:, 2:-2, :]

    # u_h = torch.fft.fft(u, dim=2)
    # # Wavenumbers in y-direction
    # k_max = nx//2
    # k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=u.device),
    #                  torch.arange(start=-k_max, end=0, step=1, device=u.device)), 0).reshape(1,1,nx)
    # ux_h = 2j *np.pi*k_x*u_h
    # uxx_h = 2j *np.pi*k_x*ux_h
    # ux = torch.fft.irfft(ux_h[:, :, :k_max+1], dim=2, n=nx)
    # uxx = torch.fft.irfft(uxx_h[:, :, :k_max+1], dim=2, n=nx)
    # ut = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dt)
    Du = ut + (ux*u - v*uxx)[:,:,1:-1]
    return Du


def PINO_loss(u, u0, v):
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    u = u.reshape(batchsize, nt, nx)
    # lploss = LpLoss(size_average=True)

    index_t = torch.zeros(nx,).long()
    index_x = torch.tensor(range(nx)).long()
    boundary_u = u[:, index_x, index_t]
    loss_u = F.mse_loss(boundary_u, u0)

    Du = FDM_Burgers1(u, v)[:, :, :]
    f = torch.zeros(Du.shape, device=u.device)
    loss_f = F.mse_loss(Du, f)

    # loss_bc0 = F.mse_loss(u[:, :, 0], u[:, :, -1])
    # loss_bc1 = F.mse_loss((u[:, :, 1] - u[:, :, -1]) /
    #                       (2/(nx)), (u[:, :, 0] - u[:, :, -2])/(2/(nx)))
    return loss_u, loss_f


def PINO_loss3d(u, u0, forcing, v=1/40, t_interval=1.0):
    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)
    nt = u.size(3)

    u = u.reshape(batchsize, nx, ny, nt)
    lploss = LpLoss(size_average=True)

    u_in = u[:, :, :, 0]
    loss_ic = lploss(u_in, u0)

    Du = FDM_NS_vorticity(u, v, t_interval)
    f = forcing.repeat(batchsize, 1, 1, nt-2)
    loss_f = lploss(Du, f)

    return loss_ic, loss_f


def PDELoss(model, x, t, nu):
    '''
    Compute the residual of PDE:
        residual = u_t + u * u_x - nu * u_{xx} : (N,1)

    Params:
        - model
        - x, t: (x, t) pairs, (N, 2) tensor
        - nu: constant of PDE
    Return:
        - mean of residual : scalar
    '''
    u = model(torch.cat([x, t], dim=1))
    # First backward to compute u_x (shape: N x 1), u_t (shape: N x 1)
    grad_x, grad_t = torch.autograd.grad(outputs=[u.sum()], inputs=[x, t], create_graph=True)
    # Second backward to compute u_{xx} (shape N x 1)

    gradgrad_x, = torch.autograd.grad(outputs=[grad_x.sum()], inputs=[x], create_graph=True)

    residual = grad_t + u * grad_x - nu * gradgrad_x
    return residual


def get_forcing(S):
    x1 = torch.tensor(np.linspace(0, 2*np.pi, S, endpoint=False), dtype=torch.float).reshape(S, 1).repeat(1, S)
    x2 = torch.tensor(np.linspace(0, 2*np.pi, S, endpoint=False), dtype=torch.float).reshape(1, S).repeat(S, 1)
    return -4 * (torch.cos(4*(x2))).reshape(1,S,S,1)

def FDM_AC(u,D=1):
    # batchsize = u.size(0)
    # nx = u.size(1)
    # ny = u.size(2)
    nx = u.size(0)
    ny = u.size(1)

    u = u.reshape(nx, ny)
    dx = D / (nx-1)
    dy = D / (ny-1)

    # ux: (batch, size-2, size-2)
    ux = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dx)
    uxx =(ux[2:, 1:-1] - ux[:-2, 1:-1]) / (2 * dx)
    uy = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dy)
    uyy = (uy[1:-1, 2:] - uy[1:-1, :-2]) / (2 * dy)
    u = u[2:-2, 2:-2]

    return u,uxx,uyy


def AC_loss(up,usol,ut,ut1):
    loss_u = F.mse_loss(up, usol, reduction = 'sum')

    ut,u_xx,u_yy = FDM_AC(ut)
    ut1,u_xx1,u_yy1 =FDM_AC(ut1)
    Du = ut1-ut - 0.5*0.02*((0.01*(u_xx+u_yy) + ut-ut**3)+(0.01*(u_xx1+u_yy1) + ut1-ut1**3))
    f = torch.zeros(Du.shape, device=ut.device)
    loss_f = F.mse_loss(Du, f)

    loss_val = loss_u + 4*loss_f
    return loss_val

def FDM_DR(s, ux, D=1):
    batchsize = s.size(0)
    nt = s.size(1)
    nx = s.size(2)

    s = s.reshape(batchsize, nt, nx)
    dt = D / (nt-1)
    dx = D / (nx-1)

    st = (s[:, :, 2:] - s[:, :, :-2]) / (2 * dt)
    sx = (s[:, 2:, :] - s[:, :-2, :]) / (2 * dx)
    sxx =(sx[:,2:, :] - sx[:, :-2, :]) / (2 * dx)
    st = st[:, 2:-2, :]
    s = s[:, 2:-2, :]
    ux = ux.reshape(batchsize, nx,1).repeat([1, 1,nt])
    ux = ux[:,2:-2,:]
    Du = st - (0.01 * sxx + 0.01 * (s**2) + ux)[:,:,1:-1]
    return Du

def DR_loss(u, ux):
    batchsize = u.size(0)
    nx = u.size(1)
    nt = u.size(2)

    u = u.reshape(batchsize, nx, nt)
    b1 = torch.zeros(u[:,0,:].shape, device=u.device)
    b2 = torch.zeros(u[:,-1,:].shape, device=u.device)
    b3 = torch.zeros(u[:,:,0].shape, device=u.device)
    loss_u = F.mse_loss(u[:,0,:], b1)+F.mse_loss(u[:,-1,:], b2)+F.mse_loss(u[:,:,0], b3)

    Du = FDM_DR(u,ux)
    f = torch.zeros(Du.shape, device=u.device)
    loss_f =  F.mse_loss(Du, f)
    return loss_u,loss_f

def FDM_AC3d(u,D=1,v=0.01):
    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)
    nt = u.size(3)

    dx = D / (nx-1)
    dy = D / (ny-1)
    dt = 0.02
    ux = (u[:,2:, 1:-1] - u[:,:-2, 1:-1]) / (2 * dx)
    uxx =(ux[:,2:, 1:-1] - ux[:,2:, 1:-1]) / (2 * dx)
    uy = (u[:,1:-1, 2:] - u[:,1:-1, :-2]) / (2 * dy)
    uyy = (uy[:,1:-1, 2:] - uy[:,1:-1, 2:]) / (2 * dy)
    ut = (u[:,:,:,2:] - u[:,:,:,:-2]) / (2 * dt)

    ut = ut[:, 2:-2, 2:-2]
    u = u[:, 2:-2, 2:-2]
    Du = ut-(v*(uxx+uyy)+u-u**3)[...,1:-1]
    return Du

def AC_loss3d(u,y):
    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)
    nt = u.size(3)

    u = u.reshape(batchsize, nx, ny, nt)
    lploss = LpLoss(size_average=True)
    # bound condition 1
    tp_u = u[:,0,:][:,None]
    tp_usol = y[:,0,:][:,None]       
    lt_u =  u[:,:,0][:,None] #L2
    lt_usol = y[:,:,0][:,None] #L2
    rt_u = u[:,:,-1][:,None]
    rt_usol = y[:,:,-1][:,None]
    
    all_u_train = torch.vstack([tp_u,lt_u,rt_u]) # X_u_train [200,2] (800 = 200(L1)+200(L2)+200(L3)+200(L4))
    all_u_sol = torch.vstack([tp_usol,lt_usol,rt_usol]) 
    # u_in = u[:, :, :, 0]
    loss_ic =F.mse_loss(all_u_train, all_u_sol)

    Du = FDM_AC3d(u)
    f = torch.zeros(Du.shape, device=u.device)
    loss_f = F.mse_loss(Du, f)
    # loss_bc0 = F.mse_loss(u[:, :, 0], u[:, :, -1])
    # loss_bc1 = F.mse_loss((u[:, :, 1] - u[:, :, -1]) /
    #                       (2/(nx)), (u[:, :, 0] - u[:, :, -2])/(2/(nx)))
    return loss_ic,loss_f
