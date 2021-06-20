
import math
import torch
import torch.nn as nn

class PRT(nn.Module):
    def __init__(self):
        super(PRT, self).__init__()
        self.device=torch.device("cuda:0")

    def factratio(self,N, D):
        if N >= D:
            prod =torch.tensor([1.0],device=self.device)
            for i in range(D + 1, N + 1):
                prod *= i
            return prod
        else:
            prod = torch.tensor([1.0], device=self.device)
            for i in range(N + 1, D + 1):
                prod *= i
            return 1.0 / prod

    def KVal(self,M, L):
        return torch.sqrt(((2 * torch.tensor([L],device=self.device) + 1) / (4 * torch.tensor([3.1415926],device=self.device))) *
                          (self.factratio(L - M, L + M)))

    def AssociatedLegendre(self,M, L, x):
        if M < 0 or M > L or torch.max(torch.abs(x)) > 1.0:
            return torch.zeros_like(x,device=self.device)

        pmm = torch.ones_like(x,device=self.device)
        if M > 0:
            somx2 = torch.sqrt((1.0 + x) * (1.0 - x))
            fact = 1.0
            for i in range(1, M + 1):
                pmm = -pmm * fact * somx2
                fact = fact + 2

        if L == M:
            return pmm
        else:
            pmmp1 = x * (2 * M + 1) * pmm
            if L == M + 1:
                return pmmp1
            else:
                pll = torch.zeros_like(x,device=self.device)
                for i in range(M + 2, L + 1):
                    pll = (x * (2 * i - 1) * pmmp1 - (i + M - 1) * pmm) / (i - M)
                    pmm = pmmp1
                    pmmp1 = pll
                return pll

    def SphericalHarmonic(self,M, L, theta, phi):
        if M > 0:
            return torch.sqrt(torch.tensor([2.0],device=self.device)) * self.KVal(M, L) * torch.cos(M * phi) * self.AssociatedLegendre(M, L, torch.cos(theta))
        elif M < 0:
            return torch.sqrt(torch.tensor([2.0],device=self.device)) * self.KVal(-M, L) * torch.sin(-M * phi) * self.AssociatedLegendre(-M, L, torch.cos(theta))
        else:
            return self.KVal(0, L) * self.AssociatedLegendre(0, L, torch.cos(theta))

    def sampleSphericalDirections(self,n):
        # Sd, directions number=n*n
        xv = torch.rand(n, n,device=self.device)
        yv = torch.rand(n, n,device=self.device)
        theta = torch.acos(1 - 2 * xv)  # range 0-2*pi
        phi = 2.0 * math.pi * yv

        phi = phi.reshape(-1)
        theta = theta.reshape(-1)

        vx = -torch.sin(theta) * torch.cos(phi)
        vy = -torch.sin(theta) * torch.sin(phi)
        vz = torch.cos(theta)
        return torch.cat([vx[:,None], vy[:,None], vz[:,None]], 1), phi, theta

    def sampleSphericalDirections_N_rays(self,N_rays,n):
        # Sd, directions number=n*n
        xv = torch.rand(N_rays,n, n,device=self.device)
        yv = torch.rand(N_rays,n, n,device=self.device)
        theta = torch.acos(1 - 2 * xv)  # range 0-2*pi
        phi = 2.0 * math.pi * yv

        phi = phi.reshape(N_rays,n*n)
        theta = theta.reshape(N_rays,n*n)

        vx = -torch.sin(theta) * torch.cos(phi)
        vy = -torch.sin(theta) * torch.sin(phi)
        vz = torch.cos(theta)
        return torch.cat([vx[:,:,None], vy[:,:,None], vz[:,:,None]], 2), phi, theta

    def getSHCoeffs(self,order, phi, theta):
        shs = []
        for n in range(0, order + 1):
            for m in range(-n, n + 1):
                s = self.SphericalHarmonic(m, n, theta, phi)
                shs.append(s[:,:,None])
        return torch.cat(shs,2)

    def computePRT_vis(self,n,order,p_vis, phi, theta):
        # vectors_orig, phi, theta = self.sampleSphericalDirections(n)  # [n*n,3]
        SH_orig = self.getSHCoeffs(order, phi, theta)  # [N_rays,n*n,9]

        w = (4.0 * torch.tensor([math.pi],device=self.device) / (n * n))

        PRT=SH_orig*p_vis
        PRT=torch.sum(PRT,dim=1) #[N_rays,9]

        PRT=torch.clamp(PRT*w,-2,2)

        return PRT

if __name__=="__main__":
    a=PRT()
    directions_, theta, phi = a.sampleSphericalDirections(20)  # [N,3]
    sh_orig=a.getSHCoeffs(2,theta,phi)
    print(sh_orig.shape)
    for i in range(10):
        print(sh_orig[i])

