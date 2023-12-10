#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param){
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++){
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }
                                                                        
            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }
                                                                        
            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }
                                                                        
            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
                                                                        
            
            
        }  // end of subcycling
    } // end of one particle
                                                                        
    return(0); // exit succcesfully
} // end of the mover

int getFlatIndex(int x, int y, int z, int maxy, int maxz){
    return x*maxy*maxz + y*maxz + z;
}

__global__ void mover_PC_kernel(
    struct particles* part, 
    struct EMfield* field, 
    struct grid* grd, 
    struct parameters* param, 
    FPpart* dt_sub_cycling,
    FPpart* dto2,
    FPpart* qomdt2,
    FPpart* x,
    FPpart* y,
    FPpart* z,
    FPpart* u,
    FPpart* v,
    FPpart* w,
    FPfield* Ex,
    FPfield* Ey,
    FPfield* Ez,
    FPfield* Bxn,
    FPfield* Byn,
    FPfield* Bzn,
    FPfield* XN,
    FPfield* YN,
    FPfield* ZN
    ){

    // get the thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= part->nop) return;

    FPpart omdtsq, denom, ut, vt, wt, udotb;

    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;

    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];

    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    xptilde = x[i];
    yptilde = y[i];
    zptilde = z[i];

    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // calculate the average velocity iteratively
        for(int innter=0; innter < part->NiterMover; innter++){
            // interpolation G-->P
            ix = 2 +  int((x[i] - grd->xStart)*grd->invdx);
            iy = 2 +  int((y[i] - grd->yStart)*grd->invdy);
            iz = 2 +  int((z[i] - grd->zStart)*grd->invdz);
            
            // calculate weights
            xi[0]   = x[i] - XN[(ix - 1) * grd->nyn * grd->nzn + iy * grd->nzn + iz];
            eta[0]  = y[i] - YN[ix * grd->nyn * grd->nzn + (iy - 1) * grd->nzn + iz];
            zeta[0] = z[i] - ZN[ix * grd->nyn * grd->nzn + iy * grd->nzn + (iz - 1)];
            xi[1]   = XN[ix * grd->nyn * grd->nzn + iy * grd->nzn + iz] - x[i];
            eta[1]  = YN[ix * grd->nyn * grd->nzn + iy * grd->nzn + iz] - y[i];
            zeta[1] = ZN[ix * grd->nyn * grd->nzn + iy * grd->nzn + iz] - z[i];
            for (int ii = 0; ii < 2; ii++)
                for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++)
                        weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
            
            // set to zero local electric and magnetic field
            Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
            
            for (int ii=0; ii < 2; ii++)
                for (int jj=0; jj < 2; jj++)
                    for(int kk=0; kk < 2; kk++){
                        Exl += weight[ii][jj][kk]*Ex[(ix - ii) * grd->nyn * grd->nzn + (iy - jj) * grd->nzn + (iz - kk)];
                        Eyl += weight[ii][jj][kk]*Ey[(ix - ii) * grd->nyn * grd->nzn + (iy - jj) * grd->nzn + (iz - kk)];
                        Ezl += weight[ii][jj][kk]*Ez[(ix - ii) * grd->nyn * grd->nzn + (iy - jj) * grd->nzn + (iz - kk)];
                        Bxl += weight[ii][jj][kk]*Bxn[(ix - ii) * grd->nyn * grd->nzn + (iy - jj) * grd->nzn + (iz - kk)];
                        Byl += weight[ii][jj][kk]*Byn[(ix - ii) * grd->nyn * grd->nzn + (iy - jj) * grd->nzn + (iz - kk)];
                        Bzl += weight[ii][jj][kk]*Bzn[(ix - ii) * grd->nyn * grd->nzn + (iy - jj) * grd->nzn + (iz - kk)];
                    }
            
            // end interpolation
            omdtsq = *qomdt2**qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
            denom = 1.0/(1.0 + omdtsq);
            // solve the position equation
            ut= u[i] + *qomdt2*Exl;
            vt= v[i] + *qomdt2*Eyl;
            wt= w[i] + *qomdt2*Ezl;
            udotb = ut*Bxl + vt*Byl + wt*Bzl;
            // solve the velocity equation
            uptilde = (ut+*qomdt2*(vt*Bzl -wt*Byl + *qomdt2*udotb*Bxl))*denom;
            vptilde = (vt+*qomdt2*(wt*Bxl -ut*Bzl + *qomdt2*udotb*Byl))*denom;
            wptilde = (wt+*qomdt2*(ut*Byl -vt*Bxl + *qomdt2*udotb*Bzl))*denom;
            // update position
            x[i] = xptilde + uptilde**dto2;
            y[i] = yptilde + vptilde**dto2;
            z[i] = zptilde + wptilde**dto2;
            
            
        } // end of iteration
        // update the final position and velocity
        u[i]= 2.0*uptilde - u[i];
        v[i]= 2.0*vptilde - v[i];
        w[i]= 2.0*wptilde - w[i];
        x[i] = xptilde + uptilde**dt_sub_cycling;
        y[i] = yptilde + vptilde**dt_sub_cycling;
        z[i] = zptilde + wptilde**dt_sub_cycling;
        
        
        //////////
        //////////
        ////////// BC
                                    
        // X-DIRECTION: BC particles
        if (x[i] > grd->Lx){
            if (param->PERIODICX==true){ // PERIODIC
                x[i] = x[i] - grd->Lx;
            } else { // REFLECTING BC
                u[i] = -u[i];
                x[i] = 2*grd->Lx - x[i];
            }
        }
                                                                    
        if (x[i] < 0){
            if (param->PERIODICX==true){ // PERIODIC
            x[i] = x[i] + grd->Lx;
            } else { // REFLECTING BC
                u[i] = -u[i];
                x[i] = -x[i];
            }
        }
            
        // Y-DIRECTION: BC particles
        if (y[i] > grd->Ly){
            if (param->PERIODICY==true){ // PERIODIC
                y[i] = y[i] - grd->Ly;
            } else { // REFLECTING BC
                v[i] = -v[i];
                y[i] = 2*grd->Ly - y[i];
            }
        }
                                                                    
        if (y[i] < 0){
            if (param->PERIODICY==true){ // PERIODIC
                y[i] = y[i] + grd->Ly;
            } else { // REFLECTING BC
                v[i] = -v[i];
                y[i] = -y[i];
            }
        }
                                                                    
        // Z-DIRECTION: BC particles
        if (z[i] > grd->Lz){
            if (param->PERIODICZ==true){ // PERIODIC
                z[i] = z[i] - grd->Lz;
            } else { // REFLECTING BC
                w[i] = -w[i];
                z[i] = 2*grd->Lz - z[i];
            }
        }
                                                                    
        if (z[i] < 0){
            if (param->PERIODICZ==true){ // PERIODIC
                z[i] = z[i] + grd->Lz;
            } else { // REFLECTING BC
                w[i] = -w[i];
                z[i] = -z[i];
            }
        }
    } // end of subcycling

}

/** particle mover gpu */
int mover_PC_gpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    // declare device variables
    struct particles *device_part;
    struct EMfield *device_field;
    struct grid *device_grd;
    struct parameters *device_param;

    FPpart *device_dt_sub_cycling;
    FPpart *device_dto2;
    FPpart *device_qomdt2;

    FPpart *device_x;
    FPpart *device_y;
    FPpart *device_z;
    FPpart *device_u;
    FPpart *device_v;
    FPpart *device_w;

    FPfield *device_Ex;
    FPfield *device_Ey;
    FPfield *device_Ez;
    FPfield *device_Bxn;
    FPfield *device_Byn;
    FPfield *device_Bzn;

    FPfield *device_xn;
    FPfield *device_yn;
    FPfield *device_zn;

    // allocate memory for device
    cudaMalloc(&device_part, sizeof(struct particles));
    cudaMalloc(&device_field, sizeof(struct EMfield));
    cudaMalloc(&device_grd, sizeof(struct grid));
    cudaMalloc(&device_param, sizeof(struct parameters));

    cudaMalloc(&device_dt_sub_cycling, sizeof(FPpart));
    cudaMalloc(&device_dto2, sizeof(FPpart));
    cudaMalloc(&device_qomdt2, sizeof(FPpart));

    int size = grd->nxn*grd->nyn*grd->nzn;

    cudaMalloc(&device_x, sizeof(FPpart)*part->npmax);
    cudaMalloc(&device_y, sizeof(FPpart)*part->npmax);
    cudaMalloc(&device_z, sizeof(FPpart)*part->npmax);
    cudaMalloc(&device_u, sizeof(FPpart)*part->npmax);
    cudaMalloc(&device_v, sizeof(FPpart)*part->npmax);
    cudaMalloc(&device_w, sizeof(FPpart)*part->npmax);

    cudaMalloc(&device_Ex, sizeof(FPfield)*size);
    cudaMalloc(&device_Ey, sizeof(FPfield)*size);
    cudaMalloc(&device_Ez, sizeof(FPfield)*size);
    cudaMalloc(&device_Bxn, sizeof(FPfield)*size);
    cudaMalloc(&device_Byn, sizeof(FPfield)*size);
    cudaMalloc(&device_Bzn, sizeof(FPfield)*size);

    cudaMalloc(&device_xn, sizeof(FPfield)*size);
    cudaMalloc(&device_yn, sizeof(FPfield)*size);
    cudaMalloc(&device_zn, sizeof(FPfield)*size);

    // copy data from host to device
    cudaMemcpy(device_part, part, sizeof(struct particles), cudaMemcpyHostToDevice);
    cudaMemcpy(device_field, field, sizeof(struct EMfield), cudaMemcpyHostToDevice);
    cudaMemcpy(device_grd, grd, sizeof(struct grid), cudaMemcpyHostToDevice);
    cudaMemcpy(device_param, param, sizeof(struct parameters), cudaMemcpyHostToDevice);

    cudaMemcpy(device_dt_sub_cycling, &dt_sub_cycling, sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(device_dto2, &dto2, sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(device_qomdt2, &qomdt2, sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMemcpy(device_x, part->x, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, part->y, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);
    cudaMemcpy(device_z, part->z, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);
    cudaMemcpy(device_u, part->u, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);
    cudaMemcpy(device_v, part->v, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);
    cudaMemcpy(device_w, part->w, sizeof(FPpart)*part->npmax, cudaMemcpyHostToDevice);

    cudaMemcpy(device_Ex, field->Ex_flat, sizeof(FPfield)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_Ey, field->Ey_flat, sizeof(FPfield)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_Ez, field->Ez_flat, sizeof(FPfield)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_Bxn, field->Bxn_flat, sizeof(FPfield)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_Byn, field->Byn_flat, sizeof(FPfield)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_Bzn, field->Bzn_flat, sizeof(FPfield)*size, cudaMemcpyHostToDevice);

    cudaMemcpy(device_xn, grd->XN_flat, sizeof(FPfield)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_yn, grd->YN_flat, sizeof(FPfield)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_zn, grd->ZN_flat, sizeof(FPfield)*size, cudaMemcpyHostToDevice);

    // set the number of threads and blocks
    int threadPerBlock = 1024;
    int blocksPerGrid = (part->nop + threadPerBlock - 1)/threadPerBlock;

    dim3 dimBlock(threadPerBlock);
    dim3 dimGrid(blocksPerGrid);

    // move each particle with new fields on device
    mover_PC_kernel<<<dimGrid, dimBlock>>>(
        device_part,
        device_field,
        device_grd,
        device_param,
        device_dt_sub_cycling, 
        device_dto2, 
        device_qomdt2,
        device_x,
        device_y,
        device_z,
        device_u,
        device_v,
        device_w,
        device_Ex,
        device_Ey,
        device_Ez,
        device_Bxn,
        device_Byn,
        device_Bzn,
        device_xn,
        device_yn,
        device_zn);

    // copy data from device to host
    cudaMemcpy(part, device_part, sizeof(struct particles), cudaMemcpyDeviceToHost);
    cudaMemcpy(field, device_field, sizeof(struct EMfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(grd, device_grd, sizeof(struct grid), cudaMemcpyDeviceToHost);
    cudaMemcpy(param, device_param, sizeof(struct parameters), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->x, device_x, sizeof(FPpart)*part->nop, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->y, device_y, sizeof(FPpart)*part->nop, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->z, device_z, sizeof(FPpart)*part->nop, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->u, device_u, sizeof(FPpart)*part->nop, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->v, device_v, sizeof(FPpart)*part->nop, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->w, device_w, sizeof(FPpart)*part->nop, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(device_part);
    cudaFree(device_field);
    cudaFree(device_grd);
    cudaFree(device_param);
    cudaFree(device_dt_sub_cycling);
    cudaFree(device_dto2);
    cudaFree(device_qomdt2);
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_z);
    cudaFree(device_u);
    cudaFree(device_v);
    cudaFree(device_w);
    cudaFree(device_Ex);
    cudaFree(device_Ey);
    cudaFree(device_Ez);
    cudaFree(device_Bxn);
    cudaFree(device_Byn);
    cudaFree(device_Bzn);
    cudaFree(device_xn);
    cudaFree(device_yn);
    cudaFree(device_zn);
                                                                        
    return(0); // exit succcesfully
}

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
       
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}

