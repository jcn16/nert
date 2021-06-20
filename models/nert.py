import torch
from torch import nn
from torch.autograd import Variable

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)

class Visibility(nn.Module):
    def __init__(self,D_1=8,D_2=4,W_1=256,W_2=128,
                 in_channels_xyz=63, in_channels_dir=27,
                 skips=[4]
                 ):
        """
        D_1: number of layers for density (sigma) encoder
        W_1: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*7*2=45 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(Visibility, self).__init__()
        self.D_1 = D_1
        self.D_2 = D_2
        self.W_1 = W_1
        self.W_2 = W_2
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        # xyz encoding layers
        for i in range(D_1):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W_1)
            elif i in skips:
                layer = nn.Linear(W_1 + in_channels_xyz, W_1)
            else:
                layer = nn.Linear(W_1, W_1)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i + 1}", layer)

        # direction encoding layers
        for i in range(D_2):
            if i == 0:
                layer = nn.Linear(in_channels_dir+W_1, W_2)
            else:
                layer = nn.Linear(W_2, W_2)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"direction_encoding_{i + 1}", layer)

        # output layers
        self.visibility = nn.Sequential(nn.Linear(W_2, 1),nn.Sigmoid())

    def forward(self, input_xyz, input_dir):
        """
        Encodes input (xyz+dir) to visibility (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction

        Outputs:
                visibility: (B, 1)
        """

        xyz_ = input_xyz
        for i in range(self.D_1):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i + 1}")(xyz_)

        mid_input=torch.cat([xyz_,input_dir],dim=-1)

        for i in range(self.D_2):
            if i==0:
                xyz_=getattr(self, f"direction_encoding_{i + 1}")(mid_input)
            else:
                xyz_ = getattr(self, f"direction_encoding_{i + 1}")(xyz_)

        out=self.visibility(xyz_)

        return out

class NeRF_sigma(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63,
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF_sigma, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # output layers
        self.sigma = nn.Linear(W, 1)

    def forward(self, x, vis_predict=False):
        xyz_ = x
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([x, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if vis_predict:
            return sigma
        else:
            return sigma,xyz_

class NeRF_SmoothTransport(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63,
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF_SmoothTransport, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # output layers
        self.sigma = nn.Linear(W, 9)

    def forward(self, x, vis_predict=False):
        xyz_ = x
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([x, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        transport = self.sigma(xyz_)
        return transport

class NeRF_SmoothNormal(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63,
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF_SmoothNormal, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # output layers
        self.sigma = nn.Linear(W, 3)

    def forward(self, x):
        xyz_ = x
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([x, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        transport = self.sigma(xyz_)
        return transport

class NeRF_SmoothAlbedo(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63+9,
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF_SmoothAlbedo, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.skips = skips
        array = [1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.sh = torch.nn.Parameter(torch.tensor(array), requires_grad=True)

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # output layers
        self.sigma = nn.Linear(W, 3)

    def forward(self, x):
        xyz_ = torch.cat([x, self.sh[None, :].expand(x.shape[0], 9)], dim=1)
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([x, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        albedo = self.sigma(xyz_)
        return albedo


class NeRF_albedo(nn.Module):
    def __init__(self, W=256, in_channels_dir=27,):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF_albedo, self).__init__()
        self.W=W
        self.in_channels_dir = in_channels_dir

        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W//2),
                                nn.ReLU(True))

        # output layers
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 3),
                        nn.Sigmoid())

    def forward(self, input_dir, features):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        xyz_encoding_final = self.xyz_encoding_final(features)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        return rgb

class NeRF_albedo_light(nn.Module):
    def __init__(self, W=256):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF_albedo_light, self).__init__()
        self.W=W
        # array=[[1.0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
        #        [1.0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
        #        [1.0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]]
        array=[1.0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
        self.sh=torch.nn.Parameter(torch.tensor(array),requires_grad=True)
        self.sh_encoding=nn.Linear(9,W//2)
        self.xyz_encoding_final = nn.Linear(W+W//2, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W, W//2),
                                nn.ReLU(True))

        # output layers
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 3),
                        nn.Sigmoid())

    def forward(self, features):
        sh_encoding=self.sh_encoding(self.sh)
        xyz_encoding_final = self.xyz_encoding_final(torch.cat([features,sh_encoding[None,:].expand(features.shape[0],self.W//2)],dim=1))
        dir_encoding = self.dir_encoding(xyz_encoding_final)
        rgb = self.rgb(dir_encoding)

        return rgb

class Transport(nn.Module):
    def __init__(self, W=256, in_channels_dir=27,):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF_albedo, self).__init__()
        self.W=W
        self.in_channels_dir = in_channels_dir

        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W),
                                nn.ReLU(True))

        self.mid=nn.Sequential(nn.Linear(W, W),
                               nn.ReLU(True),
                               nn.Linear(W, W//2),
                               nn.ReLU(True),
                               )

        # output layers
        self.transport = nn.Sequential(
                        nn.Linear(W//2, 9),
                        nn.Tanh())

    def forward(self, input_dir, features):
        xyz_encoding_final = self.xyz_encoding_final(features)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        mid=self.mid(dir_encoding)
        transport = self.transport(mid)

        return transport

class Surface_res(nn.Module):
    def __init__(self,D_1=4,W_1=256,
                 in_channels_xyz=27, in_channels_dir=63
                 ):
        super(Surface_res, self).__init__()
        self.D_1 = D_1
        self.W_1 = W_1
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir

        # xyz encoding layers

        for i in range(D_1):
            if i == 0:
                layer = nn.Linear(in_channels_xyz+in_channels_dir, W_1)
                # nn.init.normal_(layer.weight, 0.0,0.002)
                # nn.init.normal_(layer.bias, 0.0,0.002)
            else:
                layer = nn.Linear(W_1, W_1)
                # nn.init.normal_(layer.weight, 0.0,0.002)
                # nn.init.normal_(layer.bias, 0.0, 0.002)

            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i + 1}", layer)

        # output layers, output
        self.vis=nn.Linear(W_1, 1)
        # nn.init.normal_(self.vis.weight, 0.0, 0.002)
        # nn.init.normal_(self.vis.bias, 0.0,0.002)
        self.visibility = nn.Sequential(self.vis,nn.Tanh())

    def forward(self, input_xyz, input_dir):
        inputs = torch.cat([input_xyz,input_dir],dim=-1)
        xyz_=inputs
        for i in range(self.D_1):
            xyz_ = getattr(self, f"xyz_encoding_{i + 1}")(xyz_)

        out=self.visibility(xyz_)

        return out

class NeRF(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63,
                 skips=[4]):
        """
        This network has NO direction input, only input xyz, output albedo and sigma

        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.skips = skips
        self.sh=torch.nn.Parameter(torch.rand(9),requires_grad=True)
        # self.register_parameter('sh_',self.sh)
        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)

            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.sh_encoding=nn.Linear(9,W//2)
        self.xyz_encoding_final = nn.Linear(W+W//2, W)
        # output layers
        self.sigma = nn.Linear(W, 1)
        self.dir_encoding = nn.Sequential(
            nn.Linear(W, W // 2),
            nn.ReLU(True))
        self.albedo = nn.Sequential(
                        nn.Linear(W//2, 3),
                        nn.Sigmoid())

    def forward(self, x,sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:

                out: (B, 4), albedo and sigma
        """
        # print('input_shape',x.shape)
        input_xyz = x
        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([x, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        sh_encoding=self.sh_encoding(self.sh)
        xyz_=torch.cat([xyz_,sh_encoding[None,:].expand(sigma.shape[0],self.W//2)],dim=1)
        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        dir_encoding = self.dir_encoding(xyz_encoding_final)
        albedo = self.albedo(dir_encoding)

        albedo=torch.clamp(albedo,0,1)
        # self.sh=torch.clamp(self.sh,-2,2)

        out = torch.cat([albedo, sigma], -1)

        return out



