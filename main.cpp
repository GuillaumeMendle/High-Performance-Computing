#include "boost/date_time/posix_time/posix_time.hpp"
#include "cblas.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mpi.h>
using namespace std;

void transpose(double *A, int n, int m, double *B) {

  for (int v = 0; v < n; v = v + 1) {
    for (int b = 0; b < m; b = b + 1) {
      B[b * n + v] = A[v * m + b];
    }
  }
}

void multiply_mat_man(double *A, double *B, double *C, int m, int n, int kk) {

  for (int o = 0; o < m; o = o + 1) {
    for (int x = 0; x < n; x = x + 1) {
      C[o * n + x] = 0;
      for (int k = 0; k < kk; k++) {
        C[o * n + x] = C[o * n + x] + A[o * kk + k] * B[k * n + x];
      }
    }
  }
}

void multiply_vect_man(double *A, double *X, double *Y, int m, int n) {

  for (int o = 0; o < m; o = o + 1) {
    Y[o] = 0;

    for (int x = 0; x < n; x = x + 1) {

      Y[o] = Y[o] + A[o * n + x] * X[x];
    }
  }
}

void multiply(int n, double *A, double *B, double *C) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n, B,
              n, 0.0, C, n);
}

extern "C" {
// LU decomoposition
void dgetrf_(int *M, int *N, double *A, int *lda, int *IPIV, int *INFO);

// inverse of a matrix given its LU decomposition
void dgetri_(int *N, double *A, int *lda, int *IPIV, double *WORK, int *lwork,
             int *INFO);
}

void inverse(double *A, int N) {
  int *IPIV = new int[N + 1];
  int LWORK = N * N;
  double *WORK = new double[LWORK];
  int INFO;

  dgetrf_(&N, &N, A, &N, IPIV, &INFO);
  dgetri_(&N, A, &N, IPIV, WORK, &LWORK, &INFO);

  delete IPIV;
  delete WORK;
}

#define F77NAME(x) x##_

extern "C" {

void F77NAME(dgemm)(const char &trans, const char &transb, const int &m,
                    const int &n, const int &k, const double &alpha,
                    const double *A, const int &lda, const double *B,
                    const int &ldb, const double &beta, double *J,
                    const int &ldj);
}

#define F77NAME(x) x##_

extern "C" {

// LAPACK routine for solving systems of linear equations

void F77NAME(dgesv)(const int &n, const int &nrhs, const double *A,

                    const int &lda, int *ipiv, double *B,

                    const int &ldb, int &info);
}

void zeros(double *M, int n, int m) {

  for (int i = 0; i < n; i = i + 1) {

    for (int j = 0; j < m; j = j + 1) {

      M[i * m + j] = 0;
    }
  }
}

void zeros_int(int *M, int n, int m) {

  for (int i = 0; i < n; i = i + 1) {

    for (int j = 0; j < m; j = j + 1) {

      M[i * m + j] = 0;
    }
  }
}

void zeros_vect(double *V, int n) {
  for (int i = 0; i < n; i = i + 1) {
    V[i] = 0;
  }
}

void zeros_vect_int(int *V, int n) {
  for (int i = 0; i < n; i = i + 1) {
    V[i] = 0;
  }
}

void printmat(double *M, int n, int m) {

  for (int i = 0; i < n; i = i + 1) {

    for (int j = 0; j < m; j = j + 1) {

      cout << M[i * m + j] << " ";
    }

    cout << endl;
  }
}

void printvect(double *F, int n) {

  for (int i = 0; i < n; i = i + 1) {

    cout << F[i] << " ";
    ;
  }

  cout << endl;
}

void printmat_int(int *M, int n, int m) {

  for (int i = 0; i < n; i = i + 1) {

    for (int j = 0; j < m; j = j + 1) {

      cout << M[i * m + j] << " ";
    }

    cout << endl;
  }
}

void printvect_int(int *F, int n) {

  for (int i = 0; i < n; i = i + 1) {

    cout << F[i] << " ";
  }
  cout << endl;
}

int main(int argc, char *argv[])

{
  double a, h1, h2, L, tp, kxx, kyy, kxy;
  int nelem_x, nelem_y, case_num, serial_or_parallel;

  if (argc != 13) {
    cout << "Give 12 arguments" << endl;
  }

  a = atof(argv[1]);
  h1 = atof(argv[2]);
  h2 = atof(argv[3]);
  L = atof(argv[4]);
  tp = atof(argv[5]);
  kxx = atof(argv[6]);
  kyy = atof(argv[7]);
  kxy = atof(argv[8]);
  nelem_x = atoi(argv[9]);
  nelem_y = atoi(argv[10]);
  case_num = atoi(argv[11]);
  serial_or_parallel = atoi(argv[12]); // 0 for serial, 1 for parallel

  if ((kxx + kyy) <= pow(pow((kxx - kyy), 2) + 4 * pow(kxy, 2), 0.5)) {
    cout << "The conductivity matrix is not positive definite" << endl;
  } else {

    cout << "Geometric parameters" << endl;

    cout << "a= " << argv[1] << endl;

    cout << "h1= " << argv[2] << endl;

    cout << "h2= " << argv[3] << endl;

    cout << "L= " << argv[4] << endl;

    cout << "tp= " << argv[5] << endl;

    cout << "" << endl;

    cout << "Material Parameters" << endl;

    cout << "kxx= " << argv[6] << endl;

    cout << "kyy= " << argv[7] << endl;

    cout << "kxy= " << argv[8] << endl;

    cout << "" << endl;

    cout << "Discretisations parameters" << endl;

    cout << "Nelx= " << argv[9] << endl;

    cout << "Nely= " << argv[10] << endl;

    cout << "Which case would you like to resolve?" << endl;

    cout << "case number=" << argv[11] << endl;

    cout << "In serial (0) or parallel (1)?" << endl;

    cout << "serial_or_parallel=" << argv[12] << endl;
  }

  if (serial_or_parallel == 0) {

    //-------------------------------------------------------------------------------------------------------------------
    int nnode_elem = 4; // number of nodes in each element
    int nNodeDof[4] = {1, 1, 1,
                       1}; // number of DoF per node (1 = Temperature only)
    int neDof = 0;         // total number of DoF per element

    for (int i = 0; i < 4; i = i + 1) {
      neDof = neDof + nNodeDof[i];
    }
    double b = (-a * L) + ((h2 - h1) / L);

    // Calculatations

    int nelem = nelem_x * nelem_y;             // Total number of elements
    int nnode = (nelem_x + 1) * (nelem_y + 1); // Total number of nodes
    // Integration scheme

    int gaussorder = 2;

    //---------------------------------------------------------------------------------------------------------------------
    //----- Calculation of Nodal coordinate matrix -> Coord ---------

    int p = nelem_x + 1;
    int q = nelem_y + 1;
    double *x = new double[p];

    for (int i = 0; i <= nelem_x; i = i + 1) {
      double c = L / (double)nelem_x;
      x[i] = c * i;
    }

    double *h = new double[p];

    for (int i = 0; i <= nelem_x; i = i + 1) {
      h[i] = (a * pow(x[i], 2)) + (b * x[i]) + h1;
    }

    double *Y = new double[p * q];

    for (int colnr = 0; colnr <= nelem_x; colnr = colnr + 1) {

      double c = (h[colnr] / (double)nelem_y);
      double *col = new double[q];

      for (int j = 0; j <= nelem_y; j = j + 1) {

        col[j] = (-h[colnr] / 2) + c * j;
      }
      for (int i = 0; i <= nelem_y; i = i + 1) {

        Y[i * p + colnr] = col[i];
      }
    }

    double *Coord = new double[2 * nnode];

    zeros(Coord, nnode, 2);

    for (int colnr = 0; colnr <= nelem_x; colnr = colnr + 1) {
      int c = colnr * (nelem_y + 1);
      int f = (colnr + 1) * (nelem_y + 1);

      for (int i = c; i < f; i = i + 1) {
        Coord[i * 2 + 0] = x[colnr];
      }
      for (int n = c; n < f; n = n + 1) {
        int cc = n - c;
        Coord[n * 2 + 1] = Y[cc * p + colnr];
      }
    }

    //----- Calculation of topology matrix NodeTopo
    //-------------------------------------

    int *NodeTopo = new int[p * q];

    zeros_int(NodeTopo, q, p);

    for (int colnr = 0; colnr <= nelem_x; colnr = colnr + 1) {
      for (int i = 0; i <= nelem_y; i = i + 1) {

        NodeTopo[i * p + colnr] = (colnr) * (nelem_y + 1) + i;
      }
    }

    //----- Calculation of topology matrix ElemNode
    //------------------------------

    int *ElemNode = new int[5 * nelem]; // Element connectivity

    zeros_int(ElemNode, nelem, 5);

    int elemnr = 0;

    for (int colnr = 0; colnr < nelem_x; colnr = colnr + 1) {
      for (int rownr = 0; rownr < nelem_y; rownr = rownr + 1) {

        ElemNode[elemnr * 5 + 0] = elemnr;
        ElemNode[elemnr * 5 + 4] =
            NodeTopo[(rownr + 1) * p + colnr]; // Lower left node
        ElemNode[elemnr * 5 + 3] =
            NodeTopo[(rownr + 1) * p + (colnr + 1)]; // Lower right node
        ElemNode[elemnr * 5 + 2] =
            NodeTopo[rownr * p + (colnr + 1)]; // Upper right node
        ElemNode[elemnr * 5 + 1] =
            NodeTopo[rownr * p + colnr]; // upper left node
        elemnr = elemnr + 1;
      }
    }

    double *ElemX = new double[nnode_elem * nelem];
    double *ElemY = new double[nnode_elem * nelem];

    zeros(ElemX, nelem, nnode_elem);
    zeros(ElemY, nelem, nnode_elem);

    double *eNodes = new double[4];

    zeros_vect(eNodes, 4);

    double *eCoord = new double[2 * 4];

    zeros(eCoord, 4, 2);

    for (int i = 0; i < nelem; i = i + 1) {
      for (int g = 1; g < 5; g = g + 1) {
        int gg = g - 1;
        eNodes[gg] = ElemNode[i * 5 + g]; // Element nodes
      }

      for (int j = 0; j < nnode_elem; j = j + 1) {
        for (int v = 0; v <= 1; v = v + 1) {

          int m = eNodes[j];
          eCoord[j * 2 + v] = Coord[m * 2 + v];
        }
      }

      for (int h = 0; h < nnode_elem; h = h + 1) {
        ElemX[i * nnode_elem + h] = eCoord[h * 2 + 0];
        ElemY[i * nnode_elem + h] = eCoord[h * 2 + 1];
      }
    }

    //----- --------------------Generate global dof numbers
    //--------------------------------

    int *globDof = new int[2 * nnode]; // nDof/node, Dof number

    zeros_int(globDof, nnode, 2);

    int *Top = new int[4 * nelem];

    int nNode;
    int *globNodes = new int[4 * nelem];

    for (int j = 0; j < nelem; j = j + 1) {

      for (int s = 1; s < 5; s = s + 1) {

        int ss = s - 1;
        globNodes[j * 4 + ss] =
            ElemNode[j * 5 + s]; // Global node numbers of element nodes

        for (int k = 0; k < nnode_elem; k = k + 1) {
          nNode = ElemNode[j * 5 + (k + 1)];

          // if the already existing ndof of the present node is less than
          // the present elements ndof then replace the ndof for that node

          if (globDof[nNode * 2 + 0] < nNodeDof[k]) {
            globDof[nNode * 2 + 0] = nNodeDof[k];
          }
        }
      }
    }

    // counting the global dofs and inserting in globDof
    int nDof = 0;
    int eDof;
    for (int j = 0; j < nnode; j = j + 1) {

      eDof = globDof[j * 2 + 0];

      for (int k = 0; k < eDof; k = k + 1) {
        globDof[j * 2 + (k + 1)] = nDof;
        nDof = nDof + 1;
      }
    }

    //---------------------------------------------Assembly of global stiffness
    // matrix K ------------------------------
    //---------------------------------------------------- Gauss-points and
    // weights -----------------------------------

    int gauss = gaussorder; // Gauss order

    // Points
    double *GP = new double[2];

    GP[0] = -1 / pow(3, 0.5);
    GP[1] = 1 / pow(3, 0.5);

    // Weights

    int *W = new int[2];

    W[0] = 1;
    W[1] = 1;

    //----- Conductivity matrix D  -----------

    double *D = new double[2 * 2];

    D[0 * 2 + 0] = kxx;
    D[0 * 2 + 1] = kxy;
    D[1 * 2 + 0] = kxy;
    D[1 * 2 + 1] = kyy;

    //----------------------------------------

    double *K =
        new double[nDof * nDof]; // Initiation of global stiffness matrix
                                 // K

    zeros(K, nDof, nDof);

    for (int i = 0; i < nelem; i = i + 1) {

      // - data for element i

      for (int g = 1; g < 5; g = g + 1) {
        int gg = g - 1;
        eNodes[gg] = ElemNode[i * 5 + g]; // Element nodes
      }

      for (int j = 0; j < nnode_elem; j = j + 1) {
        for (int v = 0; v <= 1; v = v + 1) {
          int m = eNodes[j];
          eCoord[j * 2 + v] = Coord[m * 2 + v];
        }
      }

      int *gDof = new int[nnode_elem]; // used to constuct scatter matrix
      int gDofNode;
      for (int j = 0; j < nnode_elem; j = j + 1) {

        // global dof for node j
        int m = eNodes[j];
        int NoDf = nNodeDof[j] + 1;

        for (int k = 1; k < NoDf; k = k + 1) {
          int kk = k - 1;
          gDofNode = globDof[m * 2 + k];
        }

        gDof[j] = gDofNode;
      }

      //- Local stiffnessmatrix, Ke, is found
      //----- Element stiffness matrix, Ke, by Gauss integration -----------

      double *Ke = new double[nnode_elem * nnode_elem];

      zeros(Ke, nnode_elem, nnode_elem);

      double *DetJ =
          new double[gauss * gauss]; // For storing the determinants of J

      double *XX = new double[nnode_elem];
      double *YY = new double[nnode_elem];

      for (int h = 0; h < 4; h = h + 1) {
        XX[h] = eCoord[h * 2 + 0];
        YY[h] = eCoord[h * 2 + 1];
      }

      for (int ii = 0; ii < gauss; ii = ii + 1) {
        for (int jj = 0; jj < gauss; jj = jj + 1) {

          float eta = GP[ii];
          float xi = GP[jj];
          // shape functions matrix
          double *N = new double[4];
          N[0] = 0.25 * (1 - xi) * (1 - eta);
          N[1] = 0.25 * (1 + xi) * (1 - eta);
          N[2] = 0.25 * (1 + xi) * (1 + eta);
          N[3] = 0.25 * (1 - xi) * (1 + eta);
          // derivative (gradient) of the shape functions

          double *GN = new double[4 * 2];
          GN[0 * 4 + 0] = -0.25 * (1 - eta);
          GN[0 * 4 + 1] = 0.25 * (1 - eta);
          GN[0 * 4 + 2] = (1 + eta) * 0.25;
          GN[0 * 4 + 3] = -(1 + eta) * 0.25;

          GN[1 * 4 + 0] = -0.25 * (1 - xi);
          GN[1 * 4 + 1] = -0.25 * (1 + xi);
          GN[1 * 4 + 2] = 0.25 * (1 + xi);
          GN[1 * 4 + 3] = 0.25 * (1 - xi);

          double *J = new double[2 * 2];

          multiply_mat_man(GN, eCoord, J, 2, 2, 4);

          // Here we calculate the determinant manually as we manipulate a
          // square
          // matrix of size 2

          double DetJ;

          DetJ = J[0 * 2 + 0] * J[1 * 2 + 1] - J[0 * 2 + 1] * J[1 * 2 + 0];

          // We use the "inverse" function defined at the top of the script

          inverse(J, 2); // We obtain the inverse of J

          double *B = new double[4 * 2];

          multiply_mat_man(J, GN, B, 2, 4, 2);

          double *Bt = new double[2 * 4];

          transpose(B, 2, 4, Bt);

          double *dot = new double[2 * 4];

          multiply_mat_man(Bt, D, dot, 4, 2, 2);

          double *ddot = new double[4 * 4];

          multiply_mat_man(dot, B, ddot, 4, 4, 2);

          for (int o = 0; o < nnode_elem; o = o + 1) {
            for (int x = 0; x < nnode_elem; x = x + 1) {
              Ke[o * nnode_elem + x] =
                  Ke[o * nnode_elem + x] +
                  ddot[o * 4 + x] * tp * DetJ * W[ii] * W[jj];
            }
          }
        }
      }

      for (int v = 0; v < nnode_elem; v = v + 1) {
        int b = gDof[v];

        for (int d = 0; d < nnode_elem; d = d + 1) {

          int c = gDof[d];

          K[b * nDof + c] = K[b * nDof + c] + Ke[v * nnode_elem + d];
        }
      }
    }

    //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //---------------------------------------------CASE
    // 1--------------------------------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    if (case_num == 1) {

      // Compute nodal boundary flux vector --- natural B.C
      // Defined on edges
      int *fluxNodes = new int[q];

      for (int v = 0; v < q; v = v + 1) {
        fluxNodes[v] = NodeTopo[v * p + nelem_x];
      }

      int nFluxNodes = q;

      //----- Defining load ----------------------------
      int qflux = 2500; // Constant flux at right edge of the beam

      double *n_bc = new double[(nFluxNodes - 1) * 4];

      for (int v = 0; v < q - 1; v = v + 1) {
        n_bc[0 * (nFluxNodes - 1) + v] = fluxNodes[v]; // node 1

        n_bc[2 * (nFluxNodes - 1) + v] = qflux; // flux value at node 1
        n_bc[3 * (nFluxNodes - 1) + v] = qflux; // flux value at node 2
      }

      for (int v = 1; v < q; v = v + 1) {
        int vv = v - 1;
        n_bc[1 * (nFluxNodes - 1) + vv] = fluxNodes[v]; // node 2
      }

      int nbe = nFluxNodes - 1; // Number of elements with flux load

      double *Coordt = new double[nnode * 2];

      transpose(Coord, nnode, 2, Coordt);

      double *xcoord = new double[nnode]; // Nodal coordinates
      double *ycoord = new double[nnode];

      for (int v = 0; v < nnode; v = v + 1) {
        xcoord[v] = Coordt[0 * nnode + v];
        ycoord[v] = Coordt[1 * nnode + v];
      }

      double *f = new double[nDof];

      for (int v = 0; v < nDof; v = v + 1) {
        f[v] = 0;
      }

      double *n_bce = new double[2];

      for (int i = 0; i < nbe; i = i + 1) {

        double *fq = new double[2];
        zeros_vect(fq, 2); // initialize the nodal source vector

        int node1 = n_bc[0 * (nFluxNodes - 1) + i]; // first node
        int node2 = n_bc[1 * (nFluxNodes - 1) + i]; // second node

        for (int v = 2; v < 4; v = v + 1) {
          int vv = v - 2;
          n_bce[vv] = n_bc[v * (nFluxNodes - 1) + i]; // flux value at an edge
        }

        double x1 = xcoord[node1]; // x coord of the first node
        double y1 = ycoord[node1]; // y coord of the first node
        double x2 = xcoord[node2]; // x coord of the first node
        double y2 = ycoord[node2]; // y coord of the second node

        double leng =
            pow(pow((x2 - x1), 2) + pow((y2 - y1), 2), 0.5); // edge length
        double detJ = leng / 2;                              // 1D Jacobian

        // integrate in xi direction (1D integration)

        for (int j = 0; j < gauss; j = j + 1) {

          double xi = GP[j]; // 1D  shape functions in parent domain
          double xii = 0.5 * (1 - xi);
          double xiii = 0.5 * (1 + xi);
          double N[2] = {xii, xiii};

          double flux = 0;

          for (int o = 0; o < 2; o = o + 1) {

            flux = flux + N[o] * n_bce[o];
          }

          fq[0] = fq[0] + W[j] * N[0] * flux * detJ * tp;
          fq[1] = fq[1] + W[j] * N[1] * flux * detJ * tp;
        }
        //  define flux as negative integrals
        fq[0] = -fq[0];
        fq[1] = -fq[1];

        f[node1] = f[node1] + fq[0];
        f[node2] = f[node2] + fq[1];
      }

      //----- Apply boundary conditions ----------------- Essential
      // B.C.-------------------

      int *TempNodes = new int[q]; // Nodes at the left edge of the beam

      for (int v = 0; v < q; v = v + 1) {

        TempNodes[v] = NodeTopo[v * p + 0];
      }

      //------------------------------------------------

      int nTempNodes = q; // Number of nodes with temp BC

      double *BC =
          new double[2 * nTempNodes]; // initialize the nodal temperature vector

      zeros(BC, nTempNodes, 2);

      int T0 = 10; // Temperature at boundary

      for (int v = 0; v < nTempNodes; v = v + 1) {
        BC[v * 2 + 0] = TempNodes[v];
        BC[v * 2 + 1] = T0;
      }

      //---------------- Assembling global "Force" vector ------------

      double *OrgDof = new double[nDof]; //  Original DoF number

      zeros_vect(OrgDof, nDof);

      double *T = new double[nDof]; // initialize nodal temperature vector

      int rDof = nDof; // Reduced number of DOF

      int ind[nTempNodes];

      for (int v = 0; v < nTempNodes; v = v + 1) {
        ind[v] = BC[v * 2 + 0];
      }

      for (int v = 0; v < nTempNodes; v = v + 1) {
        int t = ind[v];
        OrgDof[t] = -1;
      }

      for (int v = 0; v < nTempNodes; v = v + 1) {
        int t = ind[v];
        OrgDof[t] = -1;
      }

      for (int v = 0; v < nTempNodes; v = v + 1) {
        int t = ind[v];
        T[t] = BC[v * 2 + 1];
      }

      rDof = rDof - nTempNodes;

      int RedDof[rDof];
      int counter1 = 0;

      for (int v = 0; v < nDof; v = v + 1) {
        if (OrgDof[v] == 0) {
          OrgDof[v] = counter1;
          RedDof[counter1] = v;
          counter1 = counter1 + 1;
        }
      }

      // Partition matrices

      int *mask_E = new int[nDof];

      for (int v = 0; v < nDof; v = v + 1) {
        for (int b = 0; b < q; b = b + 1) {
          float bb = TempNodes[b];
          if (v == bb) {
            mask_E[v] = 1;
            break;
          } else {
            mask_E[v] = 0;
          }
        }
      }

      // Obtention of the "True" in mask_E

      int mask_Ec = 0; // Define the size of mask_E containing "True"

      for (int v = 0; v < nDof; v = v + 1) {
        mask_Ec = mask_Ec + mask_E[v];
      }

      int *mask_EE = new int[mask_Ec]; // Gives the position of the answers
                                       // "True" in mask_E

      int co = 0;

      for (int v = 0; v < nDof; v = v + 1) {

        if (mask_E[v] == 1) {

          mask_EE[co] = v;
          co = co + 1;
        }
      }

      // Calculation of T_E
      double *T_E = new double[mask_Ec];

      for (int v = 0; v < mask_Ec; v = v + 1) {
        int bb = mask_EE[v];
        T_E[v] = T[bb];
      }

      // Obtention of the "False" in mask_E

      int fmask_Ec = 0; // Define the size of mask_E containing "False"

      for (int v = 0; v < nDof; v = v + 1) {
        if (mask_E[v] == 0) {
          fmask_Ec = fmask_Ec + 1;
        }
      }

      int *fmask_EE =
          new int[fmask_Ec]; // Gives the position of the "False" in mask_E

      int fco = 0;

      for (int v = 0; v < nDof; v = v + 1) {

        if (mask_E[v] == 0) {

          fmask_EE[fco] = v;
          fco = fco + 1;
        }
      }

      // Calculation of f_F

      double *f_F = new double[fmask_Ec];

      for (int v = 0; v < fmask_Ec; v = v + 1) {
        int bb = fmask_EE[v];
        f_F[v] = f[bb];
      }

      // Calculations of the partitions matrices

      double *K_EE = new double[mask_Ec * mask_Ec];

      zeros(K_EE, mask_Ec, mask_Ec);

      for (int v = 0; v < mask_Ec; v = v + 1) {
        int b = mask_EE[v];

        for (int d = 0; d < mask_Ec; d = d + 1) {

          int c = mask_EE[d];
          K_EE[v * mask_Ec + d] = K[b * nDof + c];
        }
      }

      //---------------------------------------------------------------------------------------------------------------------------

      double *K_FF = new double[fmask_Ec * fmask_Ec];

      zeros(K_FF, fmask_Ec, fmask_Ec);

      for (int v = 0; v < fmask_Ec; v = v + 1) {
        int b = fmask_EE[v];

        for (int d = 0; d < fmask_Ec; d = d + 1) {

          int c = fmask_EE[d];
          K_FF[v * fmask_Ec + d] = K[b * nDof + c];
        }
      }

      //---------------------------------------------------------------------------------------------

      double *K_EF = new double[fmask_Ec * mask_Ec];

      zeros(K_EF, mask_Ec, fmask_Ec);

      for (int v = 0; v < mask_Ec; v = v + 1) {
        int b = mask_EE[v];

        for (int d = 0; d < fmask_Ec; d = d + 1) {

          int c = fmask_EE[d];

          K_EF[v * fmask_Ec + d] = K[b * nDof + c];
        }
      }

      // solve for d_F

      double *rhs = new double[fmask_Ec];

      double *K_EFt = new double[mask_Ec * fmask_Ec];

      transpose(K_EF, mask_Ec, fmask_Ec,
                K_EFt); // Use of the function "transpose" to get the transpose
                        // of K_EF

      //"prod" is the product of K_EF.T and T_E

      double *prod = new double[fmask_Ec];

      for (int o = 0; o < fmask_Ec; o = o + 1) {

        prod[o] = 0;
        for (int k = 0; k < mask_Ec; k = k + 1) {
          prod[o] = prod[o] + K_EFt[o * mask_Ec + k] * T_E[k];
        }
      }

      for (int v = 0; v < fmask_Ec; v = v + 1) {
        rhs[v] = f_F[v] - prod[v];
      }

      // We resolve the linear system thanks to LAPACK

      double *A = new double[fmask_Ec * fmask_Ec];

      for (int b = 0; b < fmask_Ec; b = b + 1) {
        for (int v = 0; v < fmask_Ec; v = v + 1) {
          A[b * fmask_Ec + v] = K_FF[b * fmask_Ec + v];
        }
      }

      double *T_F = new double[fmask_Ec];

      for (int v = 0; v < fmask_Ec; v = v + 1) {
        T_F[v] = rhs[v];
      }

      int *ipiv = new int[fmask_Ec];

      int info = 0;

      F77NAME(dgesv)(fmask_Ec, 1, A, fmask_Ec, ipiv, T_F, fmask_Ec, info);

      // reconstruct the global displacement d

      for (int v = 0; v < mask_Ec; v = v + 1) {
        int bb = mask_EE[v];
        T[bb] = T_E[v];
      }

      for (int v = 0; v < fmask_Ec; v = v + 1) {
        int bb = fmask_EE[v];
        T[bb] = T_F[v];
      }

      // compute the reaction f_E

      double *f_E = new double[mask_Ec];

      double *dotK_EE = new double[mask_Ec];

      double *dotK_EF = new double[mask_Ec];

      for (int o = 0; o < mask_Ec; o = o + 1) {

        dotK_EE[o] = 0;
        for (int k = 0; k < mask_Ec; k = k + 1) {
          dotK_EE[o] = dotK_EE[o] + K_EE[o * mask_Ec + k] * T_E[k];
        }
      }

      for (int o = 0; o < mask_Ec; o = o + 1) {

        dotK_EF[o] = 0;

        for (int k = 0; k < fmask_Ec; k = k + 1) {
          dotK_EF[o] = dotK_EF[o] + K_EF[o * fmask_Ec + k] * T_F[k];
        }
      }

      for (int o = 0; o < mask_Ec; o = o + 1) {

        f_E[o] = dotK_EE[o] + dotK_EF[o];
      }

      // reconstruct the global reactions f

      for (int v = 0; v < mask_Ec; v = v + 1) {
        int bb = mask_EE[v];
        f[bb] = f_E[v];
      }

      for (int v = 0; v < fmask_Ec; v = v + 1) {
        int bb = fmask_EE[v];
        f[bb] = f_F[v];
      }

      //-------------------------------------------------------VTK
      // FILE-----------------------------------------------------

      // We finally write our results in a VTK file

      //-----------------------------------------------Create
      // points-----------------------------------------------------

      double *points = new double[3 * nnode];

      for (int v = 0; v < nnode; v = v + 1) {
        for (int b = 0; b < 3; b = b + 1) {

          if (b != 2) {
            points[v * 3 + b] = Coord[v * 2 + b];

          }

          else {
            points[v * 3 + b] = 0;
          }
        }
      }

      //--------------------------------------------Create
      // cells---------------------------------------------------------

      int *cells = new int[4 * nelem];

      for (int i = 0; i < nelem; i = i + 1) {

        for (int j = 0; j < 4; j = j + 1) {

          int jj = j + 1;

          cells[i * 4 + j] = ElemNode[i * 5 + jj];
        }
      }

      //---------------Create VTK
      // file----------------------------------------------------------------

      // Create intro

      // To avoid misunderstanding the names of the files will be "datas1",
      // "datas2"
      // and "datas3"

      ofstream vOut("datas1.vtk", ios::out | ios::trunc);
      vOut << "# vtk DataFile Version 4.0" << endl;
      vOut << "vtk output" << endl;
      vOut << "ASCII" << endl;
      vOut << "DATASET UNSTRUCTURED_GRID" << endl;

      // Print points

      vOut << "POINTS"
           << " " << nnode << " "
           << "double" << endl;
      for (int v = 0; v < nnode; v = v + 1) {
        for (int b = 0; b < 3; b = b + 1) {
          vOut << points[v * 3 + b] << " ";
        }
      }
      vOut << endl;

      // print cells

      int total_num_cells = nelem;
      int total_num_idx = 5 * nelem;

      vOut << "CELLS"
           << " " << total_num_cells << " " << total_num_idx << endl;

      // Creation of keys_cells

      int *keys_cells = new int[5 * nelem];

      for (int i = 0; i < nelem; i = i + 1) {
        keys_cells[i * 5 + 0] = 4;
        for (int j = 1; j < 5; j = j + 1) {
          int jj = j - 1;
          keys_cells[i * 5 + j] = cells[i * 4 + jj];
        }
      }

      // print keys_cells

      for (int i = 0; i < total_num_cells; i = i + 1) {

        for (int j = 0; j < 5; j = j + 1) {

          vOut << keys_cells[i * 5 + j] << " ";
        }
        vOut << endl;
      }

      vOut << "CELL_TYPES"
           << " " << total_num_cells << endl;

      for (int i = 0; i < total_num_cells; i = i + 1) {

        vOut << 9 << endl;
      }

      // Here we don't create "point_data" as we directly use T and len(T)=nDof

      // Print point_data

      int len_points = nnode;

      vOut << "POINT_DATA"
           << " " << len_points << endl;
      vOut << "FIELD FieldData"
           << " "
           << "1" << endl;
      vOut << "disp"
           << " "
           << "1"
           << " " << nDof << " "
           << "double" << endl;

      for (int i = 0; i < nDof; i = i + 1) {
        vOut << T[i] << " ";
      }

      vOut.close();
    }

    //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------CASE
    // 2-------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    else if (case_num == 2) {

      // Compute nodal boundary flux vector --- natural B.C
      // Defined on edges

      int *fluxNodes = new int[p];

      for (int v = 0; v < p; v = v + 1) {
        fluxNodes[v] = NodeTopo[nelem_y * p + v];
      }

      int nFluxNodes = p;

      //----- Defining load ----------------------------
      int qflux = 2500; // Constant flux at right edge of the beam

      double *n_bc = new double[(nFluxNodes - 1) * 4];

      for (int v = 0; v < nFluxNodes - 1; v = v + 1) {
        n_bc[0 * (nFluxNodes - 1) + v] = fluxNodes[v]; // node 1

        n_bc[2 * (nFluxNodes - 1) + v] = qflux; // flux value at node 1
        n_bc[3 * (nFluxNodes - 1) + v] = qflux; // flux value at node 2
      }

      for (int v = 1; v < nFluxNodes; v = v + 1) {
        int vv = v - 1;
        n_bc[1 * (nFluxNodes - 1) + vv] = fluxNodes[v]; // node 2
      }

      int nbe = nFluxNodes - 1; // Number of elements with flux load

      double *Coordt = new double[nnode * 2];

      transpose(Coord, nnode, 2, Coordt);

      double *xcoord = new double[nnode]; // Nodal coordinates
      double *ycoord = new double[nnode];

      for (int v = 0; v < nnode; v = v + 1) {
        xcoord[v] = Coordt[0 * nnode + v];
        ycoord[v] = Coordt[1 * nnode + v];
      }

      double *f = new double[nDof];

      for (int v = 0; v < nDof; v = v + 1) {
        f[v] = 0;
      }

      double *n_bce = new double[2];

      for (int i = 0; i < nbe; i = i + 1) {

        double *fq = new double[2];
        zeros_vect(fq, 2); // initialize the nodal source vector

        int node1 = n_bc[0 * (nFluxNodes - 1) + i]; // first node
        int node2 = n_bc[1 * (nFluxNodes - 1) + i]; // second node

        for (int v = 2; v < 4; v = v + 1) {
          int vv = v - 2;
          n_bce[vv] = n_bc[v * (nFluxNodes - 1) + i]; // flux value at an edge
        }

        double x1 = xcoord[node1]; // x coord of the first node
        double y1 = ycoord[node1]; // y coord of the first node
        double x2 = xcoord[node2]; // x coord of the first node
        double y2 = ycoord[node2]; // y coord of the second node

        double leng =
            pow(pow((x2 - x1), 2) + pow((y2 - y1), 2), 0.5); // edge length
        double detJ = leng / 2;                              // 1D Jacobian

        // integrate in xi direction (1D integration)

        for (int j = 0; j < gauss; j = j + 1) {

          double xi = GP[j]; // 1D  shape functions in parent domain
          double xii = 0.5 * (1 - xi);
          double xiii = 0.5 * (1 + xi);
          double N[2] = {xii, xiii};

          double flux = 0;

          for (int o = 0; o < 2; o = o + 1) {

            flux = flux + N[o] * n_bce[o];
          }

          fq[0] = fq[0] + W[j] * N[0] * flux * detJ * tp;
          fq[1] = fq[1] + W[j] * N[1] * flux * detJ * tp;
        }
        //  define flux as negative integrals
        fq[0] = -fq[0];
        fq[1] = -fq[1];

        f[node1] = f[node1] + fq[0];
        f[node2] = f[node2] + fq[1];
      }

      //----- Apply boundary conditions ----------------- Essential
      // B.C.-------------------

      int *TempNodes = new int[p]; // Nodes at the left edge of the beam

      for (int v = 0; v < p; v = v + 1) {

        TempNodes[v] = NodeTopo[0 * p + v];
      }

      //------------------------------------------------

      int nTempNodes = p; // Number of nodes with temp BC

      double *BC =
          new double[2 * nTempNodes]; // initialize the nodal temperature vector

      zeros(BC, nTempNodes, 2);

      int T0 = 10; // Temperature at boundary

      for (int v = 0; v < nTempNodes; v = v + 1) {
        BC[v * 2 + 0] = TempNodes[v];
        BC[v * 2 + 1] = T0;
      }

      //---------------- Assembling global "Force" vector ------------

      double *OrgDof = new double[nDof]; //  Original DoF number

      zeros_vect(OrgDof, nDof);

      double *T = new double[nDof]; // initialize nodal temperature vector

      int rDof = nDof; // Reduced number of DOF

      int ind[nTempNodes];

      for (int v = 0; v < nTempNodes; v = v + 1) {
        ind[v] = BC[v * 2 + 0];
      }

      for (int v = 0; v < nTempNodes; v = v + 1) {
        int t = ind[v];
        OrgDof[t] = -1;
      }

      for (int v = 0; v < nTempNodes; v = v + 1) {
        int t = ind[v];
        OrgDof[t] = -1;
      }

      for (int v = 0; v < nTempNodes; v = v + 1) {
        int t = ind[v];
        T[t] = BC[v * 2 + 1];
      }

      rDof = rDof - nTempNodes;

      int RedDof[rDof];
      int counter1 = 0;

      for (int v = 0; v < nDof; v = v + 1) {
        if (OrgDof[v] == 0) {
          OrgDof[v] = counter1;
          RedDof[counter1] = v;
          counter1 = counter1 + 1;
        }
      }

      // Partition matrices

      int *mask_E = new int[nDof];

      for (int v = 0; v < nDof; v = v + 1) {
        for (int b = 0; b < nTempNodes; b = b + 1) {
          float bb = TempNodes[b];
          if (v == bb) {
            mask_E[v] = 1;
            break;
          } else {
            mask_E[v] = 0;
          }
        }
      }

      //-------------------------------------------------------------------------------------------------------------------------------------------------

      int mask_Ec = 0;

      for (int v = 0; v < nDof; v = v + 1) {
        mask_Ec = mask_Ec + mask_E[v];
      }

      int *mask_EE = new int[mask_Ec];

      int co = 0;

      for (int v = 0; v < nDof; v = v + 1) {

        if (mask_E[v] == 1) {

          mask_EE[co] = v;
          co = co + 1;
        }
      }

      double *T_E = new double[mask_Ec];

      for (int v = 0; v < mask_Ec; v = v + 1) {
        int bb = mask_EE[v];
        T_E[v] = T[bb];
      }

      //--------------------------------------------------------------------------------------------------------------------------------------------

      int fmask_Ec = 0;

      for (int v = 0; v < nDof; v = v + 1) {
        if (mask_E[v] == 0) {
          fmask_Ec = fmask_Ec + 1;
        }
      }

      int *fmask_EE = new int[fmask_Ec];

      int fco = 0;

      for (int v = 0; v < nDof; v = v + 1) {

        if (mask_E[v] == 0) {

          fmask_EE[fco] = v;
          fco = fco + 1;
        }
      }

      double *f_F = new double[fmask_Ec];

      for (int v = 0; v < fmask_Ec; v = v + 1) {
        int bb = fmask_EE[v];
        f_F[v] = f[bb];
      }

      //--------------------------------------------------------------------------------------------------------

      double *K_EE = new double[mask_Ec * mask_Ec];

      zeros(K_EE, mask_Ec, mask_Ec);

      for (int v = 0; v < mask_Ec; v = v + 1) {
        int b = mask_EE[v];

        for (int d = 0; d < mask_Ec; d = d + 1) {

          int c = mask_EE[d];
          K_EE[v * mask_Ec + d] = K[b * nDof + c];
        }
      }

      //---------------------------------------------------------------------------------------------------------------------------

      double *K_FF = new double[fmask_Ec * fmask_Ec];

      zeros(K_FF, fmask_Ec, fmask_Ec);

      for (int v = 0; v < fmask_Ec; v = v + 1) {
        int b = fmask_EE[v];

        for (int d = 0; d < fmask_Ec; d = d + 1) {

          int c = fmask_EE[d];
          K_FF[v * fmask_Ec + d] = K[b * nDof + c];
        }
      }

      //---------------------------------------------------------------------------------------------

      double *K_EF = new double[fmask_Ec * mask_Ec];

      zeros(K_EF, mask_Ec, fmask_Ec);

      for (int v = 0; v < mask_Ec; v = v + 1) {
        int b = mask_EE[v];

        for (int d = 0; d < fmask_Ec; d = d + 1) {

          int c = fmask_EE[d];

          K_EF[v * fmask_Ec + d] = K[b * nDof + c];
        }
      }

      // solve for d_F

      double *rhs = new double[fmask_Ec];

      double *K_EFt = new double[mask_Ec * fmask_Ec];

      transpose(K_EF, mask_Ec, fmask_Ec, K_EFt);

      double *prod = new double[fmask_Ec];

      for (int o = 0; o < fmask_Ec; o = o + 1) {

        prod[o] = 0;
        for (int k = 0; k < mask_Ec; k = k + 1) {
          prod[o] = prod[o] + K_EFt[o * mask_Ec + k] * T_E[k];
        }
      }

      for (int v = 0; v < fmask_Ec; v = v + 1) {
        rhs[v] = f_F[v] - prod[v];
      }

      // Linear system resolution

      double *A = new double[fmask_Ec * fmask_Ec];

      for (int b = 0; b < fmask_Ec; b = b + 1) {
        for (int v = 0; v < fmask_Ec; v = v + 1) {
          A[b * fmask_Ec + v] = K_FF[b * fmask_Ec + v];
        }
      }

      double *T_F = new double[fmask_Ec];

      for (int v = 0; v < fmask_Ec; v = v + 1) {
        T_F[v] = rhs[v];
      }

      int *ipiv = new int[fmask_Ec];

      int info = 0;

      F77NAME(dgesv)(fmask_Ec, 1, A, fmask_Ec, ipiv, T_F, fmask_Ec, info);

      // reconstruct the global displacement d

      for (int v = 0; v < mask_Ec; v = v + 1) {
        int bb = mask_EE[v];
        T[bb] = T_E[v];
      }

      for (int v = 0; v < fmask_Ec; v = v + 1) {
        int bb = fmask_EE[v];
        T[bb] = T_F[v];
      }

      // compute the reaction f_E

      double *f_E = new double[mask_Ec];

      double *dotK_EE = new double[mask_Ec];

      double *dotK_EF = new double[mask_Ec];

      for (int o = 0; o < mask_Ec; o = o + 1) {

        dotK_EE[o] = 0;
        for (int k = 0; k < mask_Ec; k = k + 1) {
          dotK_EE[o] = dotK_EE[o] + K_EE[o * mask_Ec + k] * T_E[k];
        }
      }

      for (int o = 0; o < mask_Ec; o = o + 1) {

        dotK_EF[o] = 0;

        for (int k = 0; k < fmask_Ec; k = k + 1) {
          dotK_EF[o] = dotK_EF[o] + K_EF[o * fmask_Ec + k] * T_F[k];
        }
      }

      for (int o = 0; o < mask_Ec; o = o + 1) {

        f_E[o] = dotK_EE[o] + dotK_EF[o];
      }

      // reconstruct the global reactions f

      for (int v = 0; v < mask_Ec; v = v + 1) {
        int bb = mask_EE[v];
        f[bb] = f_E[v];
      }

      for (int v = 0; v < fmask_Ec; v = v + 1) {
        int bb = fmask_EE[v];
        f[bb] = f_F[v];
      }

      //-----------------------------------------------Create
      // points-----------------------------------------------------

      double *points = new double[3 * nnode];

      for (int v = 0; v < nnode; v = v + 1) {
        for (int b = 0; b < 3; b = b + 1) {

          if (b != 2) {
            points[v * 3 + b] = Coord[v * 2 + b];

          }

          else {
            points[v * 3 + b] = 0;
          }
        }
      }

      //--------------------------------------------Create
      // cells---------------------------------------------------------

      int *cells = new int[4 * nelem];

      for (int i = 0; i < nelem; i = i + 1) {

        for (int j = 0; j < 4; j = j + 1) {

          int jj = j + 1;

          cells[i * 4 + j] = ElemNode[i * 5 + jj];
        }
      }

      //---------------Create VTK
      // file----------------------------------------------------------------

      // Create intro

      ofstream vOut("datas2.vtk", ios::out | ios::trunc);
      vOut << "# vtk DataFile Version 4.0" << endl;
      vOut << "vtk output" << endl;
      vOut << "ASCII" << endl;
      vOut << "DATASET UNSTRUCTURED_GRID" << endl;

      // Print points

      vOut << "POINTS"
           << " " << nnode << " "
           << "double" << endl;
      for (int v = 0; v < nnode; v = v + 1) {
        for (int b = 0; b < 3; b = b + 1) {
          vOut << points[v * 3 + b] << " ";
        }
      }
      vOut << endl;

      // print cells

      int total_num_cells = nelem;
      int total_num_idx = 5 * nelem;

      vOut << "CELLS"
           << " " << total_num_cells << " " << total_num_idx << endl;

      // Creation of keys_cells

      int *keys_cells = new int[5 * nelem];

      for (int i = 0; i < nelem; i = i + 1) {
        keys_cells[i * 5 + 0] = 4;
        for (int j = 1; j < 5; j = j + 1) {
          int jj = j - 1;
          keys_cells[i * 5 + j] = cells[i * 4 + jj];
        }
      }

      // print keys_cells

      for (int i = 0; i < total_num_cells; i = i + 1) {

        for (int j = 0; j < 5; j = j + 1) {

          vOut << keys_cells[i * 5 + j] << " ";
        }
        vOut << endl;
      }

      vOut << "CELL_TYPES"
           << " " << total_num_cells << endl;

      for (int i = 0; i < total_num_cells; i = i + 1) {

        vOut << 9 << endl;
      }

      // Here we don't create "point_data" as we directly use T and len(T)=nDof

      // Print point_data

      int len_points = nnode;

      vOut << "POINT_DATA"
           << " " << len_points << endl;
      vOut << "FIELD FieldData"
           << " "
           << "1" << endl;
      vOut << "disp"
           << " "
           << "1"
           << " " << nDof << " "
           << "double" << endl;

      for (int i = 0; i < nDof; i = i + 1) {
        vOut << T[i] << " ";
      }

      vOut.close();
    }

    //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------CASE
    // 3-------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    else if (case_num == 3) {

      // Compute nodal boundary flux vector --- natural B.C
      // Defined on edges

      int *fluxNodes = new int[p];

      for (int v = 0; v < p; v = v + 1) {
        fluxNodes[v] = NodeTopo[0 * p + v];
      }

      int nFluxNodes = p;

      //----- Defining load ----------------------------
      int qflux = -5000; // Constant flux at right edge of the beam

      double *n_bc = new double[(nFluxNodes - 1) * 4];

      for (int v = 0; v < nFluxNodes - 1; v = v + 1) {
        n_bc[0 * (nFluxNodes - 1) + v] = fluxNodes[v]; // node 1

        n_bc[2 * (nFluxNodes - 1) + v] = qflux; // flux value at node 1
        n_bc[3 * (nFluxNodes - 1) + v] = qflux; // flux value at node 2
      }

      for (int v = 1; v < nFluxNodes; v = v + 1) {
        int vv = v - 1;
        n_bc[1 * (nFluxNodes - 1) + vv] = fluxNodes[v]; // node 2
      }

      int nbe = nFluxNodes - 1; // Number of elements with flux load

      double *Coordt = new double[nnode * 2];

      transpose(Coord, nnode, 2, Coordt);

      double *xcoord = new double[nnode]; // Nodal coordinates
      double *ycoord = new double[nnode];

      for (int v = 0; v < nnode; v = v + 1) {
        xcoord[v] = Coordt[0 * nnode + v];
        ycoord[v] = Coordt[1 * nnode + v];
      }

      double *f = new double[nDof];

      for (int v = 0; v < nDof; v = v + 1) {
        f[v] = 0;
      }

      double *n_bce = new double[2];

      for (int i = 0; i < nbe; i = i + 1) {

        double *fq = new double[2];
        zeros_vect(fq, 2); // initialize the nodal source vector

        int node1 = n_bc[0 * (nFluxNodes - 1) + i]; // first node
        int node2 = n_bc[1 * (nFluxNodes - 1) + i]; // second node

        for (int v = 2; v < 4; v = v + 1) {
          int vv = v - 2;
          n_bce[vv] = n_bc[v * (nFluxNodes - 1) + i]; // flux value at an edge
        }

        double x1 = xcoord[node1]; // x coord of the first node
        double y1 = ycoord[node1]; // y coord of the first node
        double x2 = xcoord[node2]; // x coord of the first node
        double y2 = ycoord[node2]; // y coord of the second node

        double leng =
            pow(pow((x2 - x1), 2) + pow((y2 - y1), 2), 0.5); // edge length
        double detJ = leng / 2;                              // 1D Jacobian

        // integrate in xi direction (1D integration)

        for (int j = 0; j < gauss; j = j + 1) {

          double xi = GP[j]; // 1D  shape functions in parent domain
          double xii = 0.5 * (1 - xi);
          double xiii = 0.5 * (1 + xi);
          double N[2] = {xii, xiii};

          double flux = 0;

          for (int o = 0; o < 2; o = o + 1) {

            flux = flux + N[o] * n_bce[o];
          }

          fq[0] = fq[0] + W[j] * N[0] * flux * detJ * tp;
          fq[1] = fq[1] + W[j] * N[1] * flux * detJ * tp;
        }
        //  define flux as negative integrals
        fq[0] = -fq[0];
        fq[1] = -fq[1];

        f[node1] = f[node1] + fq[0];
        f[node2] = f[node2] + fq[1];
      }

      //----- Apply boundary conditions ----------------- Essential
      // B.C.-------------------

      int *TempNodes = new int[q]; // Nodes at the left edge of the beam

      for (int v = 0; v < q; v = v + 1) {

        TempNodes[v] = NodeTopo[v * p + 0];
      }

      //------------------------------------------------

      int nTempNodes = q; // Number of nodes with temp BC

      double *BC =
          new double[2 * nTempNodes]; // initialize the nodal temperature vector

      zeros(BC, nTempNodes, 2);

      int T0 = -20; // Temperature at boundary

      for (int v = 0; v < nTempNodes; v = v + 1) {
        BC[v * 2 + 0] = TempNodes[v];
        BC[v * 2 + 1] = T0;
      }

      //---------------- Assembling global "Force" vector ------------

      double *OrgDof = new double[nDof]; //  Original DoF number

      zeros_vect(OrgDof, nDof);

      double *T = new double[nDof]; // initialize nodal temperature vector

      int rDof = nDof; // Reduced number of DOF

      int ind[nTempNodes];

      for (int v = 0; v < nTempNodes; v = v + 1) {
        ind[v] = BC[v * 2 + 0];
      }

      for (int v = 0; v < nTempNodes; v = v + 1) {
        int t = ind[v];
        OrgDof[t] = -1;
      }

      for (int v = 0; v < nTempNodes; v = v + 1) {
        int t = ind[v];
        OrgDof[t] = -1;
      }

      for (int v = 0; v < nTempNodes; v = v + 1) {
        int t = ind[v];
        T[t] = BC[v * 2 + 1];
      }

      rDof = rDof - nTempNodes;

      int RedDof[rDof];
      int counter1 = 0;

      for (int v = 0; v < nDof; v = v + 1) {
        if (OrgDof[v] == 0) {
          OrgDof[v] = counter1;
          RedDof[counter1] = v;
          counter1 = counter1 + 1;
        }
      }

      // Partition matrices

      int *mask_E = new int[nDof];

      for (int v = 0; v < nDof; v = v + 1) {
        for (int b = 0; b < nTempNodes; b = b + 1) {
          float bb = TempNodes[b];
          if (v == bb) {
            mask_E[v] = 1;
            break;
          } else {
            mask_E[v] = 0;
          }
        }
      }

      //-------------------------------------------------------------------------------------------------------------------------------------------------

      int mask_Ec = 0;

      for (int v = 0; v < nDof; v = v + 1) {
        mask_Ec = mask_Ec + mask_E[v];
      }

      int *mask_EE = new int[mask_Ec];

      int co = 0;

      for (int v = 0; v < nDof; v = v + 1) {

        if (mask_E[v] == 1) {

          mask_EE[co] = v;
          co = co + 1;
        }
      }

      double *T_E = new double[mask_Ec];

      for (int v = 0; v < mask_Ec; v = v + 1) {
        int bb = mask_EE[v];
        T_E[v] = T[bb];
      }

      //--------------------------------------------------------------------------------------------------------------------------------------------

      int fmask_Ec = 0;

      for (int v = 0; v < nDof; v = v + 1) {
        if (mask_E[v] == 0) {
          fmask_Ec = fmask_Ec + 1;
        }
      }

      int *fmask_EE = new int[fmask_Ec];

      int fco = 0;

      for (int v = 0; v < nDof; v = v + 1) {

        if (mask_E[v] == 0) {

          fmask_EE[fco] = v;
          fco = fco + 1;
        }
      }

      double *f_F = new double[fmask_Ec];

      for (int v = 0; v < fmask_Ec; v = v + 1) {
        int bb = fmask_EE[v];
        f_F[v] = f[bb];
      }

      //--------------------------------------------------------------------------------------------------------

      double *K_EE = new double[mask_Ec * mask_Ec];

      zeros(K_EE, mask_Ec, mask_Ec);

      for (int v = 0; v < mask_Ec; v = v + 1) {
        int b = mask_EE[v];

        for (int d = 0; d < mask_Ec; d = d + 1) {

          int c = mask_EE[d];
          K_EE[v * mask_Ec + d] = K[b * nDof + c];
        }
      }

      //---------------------------------------------------------------------------------------------------------------------------

      double *K_FF = new double[fmask_Ec * fmask_Ec];

      zeros(K_FF, fmask_Ec, fmask_Ec);

      for (int v = 0; v < fmask_Ec; v = v + 1) {
        int b = fmask_EE[v];

        for (int d = 0; d < fmask_Ec; d = d + 1) {

          int c = fmask_EE[d];
          K_FF[v * fmask_Ec + d] = K[b * nDof + c];
        }
      }

      //---------------------------------------------------------------------------------------------

      double *K_EF = new double[fmask_Ec * mask_Ec];

      zeros(K_EF, mask_Ec, fmask_Ec);

      for (int v = 0; v < mask_Ec; v = v + 1) {
        int b = mask_EE[v];

        for (int d = 0; d < fmask_Ec; d = d + 1) {

          int c = fmask_EE[d];

          K_EF[v * fmask_Ec + d] = K[b * nDof + c];
        }
      }

      // solve for d_F

      double *rhs = new double[fmask_Ec];

      double *K_EFt = new double[mask_Ec * fmask_Ec];

      transpose(K_EF, mask_Ec, fmask_Ec, K_EFt);

      double *prod = new double[fmask_Ec];

      for (int o = 0; o < fmask_Ec; o = o + 1) {

        prod[o] = 0;
        for (int k = 0; k < mask_Ec; k = k + 1) {
          prod[o] = prod[o] + K_EFt[o * mask_Ec + k] * T_E[k];
        }
      }

      for (int v = 0; v < fmask_Ec; v = v + 1) {
        rhs[v] = f_F[v] - prod[v];
      }

      double *A = new double[fmask_Ec * fmask_Ec];

      for (int b = 0; b < fmask_Ec; b = b + 1) {
        for (int v = 0; v < fmask_Ec; v = v + 1) {
          A[b * fmask_Ec + v] = K_FF[b * fmask_Ec + v];
        }
      }

      double *T_F = new double[fmask_Ec];

      for (int v = 0; v < fmask_Ec; v = v + 1) {
        T_F[v] = rhs[v];
      }

      int *ipiv = new int[fmask_Ec];

      int info = 0;

      F77NAME(dgesv)(fmask_Ec, 1, A, fmask_Ec, ipiv, T_F, fmask_Ec, info);

      // reconstruct the global displacement d

      for (int v = 0; v < mask_Ec; v = v + 1) {
        int bb = mask_EE[v];
        T[bb] = T_E[v];
      }

      for (int v = 0; v < fmask_Ec; v = v + 1) {
        int bb = fmask_EE[v];
        T[bb] = T_F[v];
      }

      // compute the reaction f_E

      double *f_E = new double[mask_Ec];

      double *dotK_EE = new double[mask_Ec];

      double *dotK_EF = new double[mask_Ec];

      for (int o = 0; o < mask_Ec; o = o + 1) {

        dotK_EE[o] = 0;
        for (int k = 0; k < mask_Ec; k = k + 1) {
          dotK_EE[o] = dotK_EE[o] + K_EE[o * mask_Ec + k] * T_E[k];
        }
      }

      for (int o = 0; o < mask_Ec; o = o + 1) {

        dotK_EF[o] = 0;

        for (int k = 0; k < fmask_Ec; k = k + 1) {
          dotK_EF[o] = dotK_EF[o] + K_EF[o * fmask_Ec + k] * T_F[k];
        }
      }

      for (int o = 0; o < mask_Ec; o = o + 1) {

        f_E[o] = dotK_EE[o] + dotK_EF[o];
      }

      // reconstruct the global reactions f

      for (int v = 0; v < mask_Ec; v = v + 1) {
        int bb = mask_EE[v];
        f[bb] = f_E[v];
      }

      for (int v = 0; v < fmask_Ec; v = v + 1) {
        int bb = fmask_EE[v];
        f[bb] = f_F[v];
      }

      //-----------------------------------------------Create
      // points-----------------------------------------------------

      double *points = new double[3 * nnode];

      for (int v = 0; v < nnode; v = v + 1) {
        for (int b = 0; b < 3; b = b + 1) {

          if (b != 2) {
            points[v * 3 + b] = Coord[v * 2 + b];

          }

          else {
            points[v * 3 + b] = 0;
          }
        }
      }

      //--------------------------------------------Create
      // cells---------------------------------------------------------

      int *cells = new int[4 * nelem];

      for (int i = 0; i < nelem; i = i + 1) {

        for (int j = 0; j < 4; j = j + 1) {

          int jj = j + 1;

          cells[i * 4 + j] = ElemNode[i * 5 + jj];
        }
      }

      //---------------Create VTK
      // file----------------------------------------------------------------

      // Create intro

      ofstream vOut("datas3.vtk", ios::out | ios::trunc);
      vOut << "# vtk DataFile Version 4.0" << endl;
      vOut << "vtk output" << endl;
      vOut << "ASCII" << endl;
      vOut << "DATASET UNSTRUCTURED_GRID" << endl;

      // Print points

      vOut << "POINTS"
           << " " << nnode << " "
           << "double" << endl;
      for (int v = 0; v < nnode; v = v + 1) {
        for (int b = 0; b < 3; b = b + 1) {
          vOut << points[v * 3 + b] << " ";
        }
      }
      vOut << endl;

      // print cells

      int total_num_cells = nelem;
      int total_num_idx = 5 * nelem;

      vOut << "CELLS"
           << " " << total_num_cells << " " << total_num_idx << endl;

      // Creation of keys_cells

      int *keys_cells = new int[5 * nelem];

      for (int i = 0; i < nelem; i = i + 1) {
        keys_cells[i * 5 + 0] = 4;
        for (int j = 1; j < 5; j = j + 1) {
          int jj = j - 1;
          keys_cells[i * 5 + j] = cells[i * 4 + jj];
        }
      }

      // print keys_cells

      for (int i = 0; i < total_num_cells; i = i + 1) {

        for (int j = 0; j < 5; j = j + 1) {

          vOut << keys_cells[i * 5 + j] << " ";
        }
        vOut << endl;
      }

      vOut << "CELL_TYPES"
           << " " << total_num_cells << endl;

      for (int i = 0; i < total_num_cells; i = i + 1) {

        vOut << 9 << endl;
      }

      // Here we don't create "point_data" as we directly use T and len(T)=nDof

      // Print point_data

      int len_points = nnode;

      vOut << "POINT_DATA"
           << " " << len_points << endl;
      vOut << "FIELD FieldData"
           << " "
           << "1" << endl;
      vOut << "disp"
           << " "
           << "1"
           << " " << nDof << " "
           << "double" << endl;

      for (int i = 0; i < nDof; i = i + 1) {
        vOut << T[i] << " ";
      }

      vOut.close();
    }
  }

  if (serial_or_parallel == 1) {

    //-------------------------------------------------------------------------------------------------------------------
    int nnode_elem = 4; // number of nodes in each element
    int nNodeDof[4] = {1, 1, 1,
                       1}; // number of DoF per node (1 = Temperature only)
    int neDof = 0;         // total number of DoF per element

    for (int i = 0; i < 4; i = i + 1) {
      neDof = neDof + nNodeDof[i];
    }
    double b = (-a * L) + ((h2 - h1) / L);

    // Calculatations

    int nelem = nelem_x * nelem_y;             // Total number of elements
    int nnode = (nelem_x + 1) * (nelem_y + 1); // Total number of nodes
    // Integration scheme

    int gaussorder = 2;

    //---------------------------------------------------------------------------------------------------------------------
    //----- Calculation of Nodal coordinate matrix -> Coord ---------

    int p = nelem_x + 1;
    int q = nelem_y + 1;
    double *x = new double[p];

    for (int i = 0; i <= nelem_x; i = i + 1) {
      double c = L / (double)nelem_x;
      x[i] = c * i;
    }

    double *h = new double[p];

    for (int i = 0; i <= nelem_x; i = i + 1) {
      h[i] = (a * pow(x[i], 2)) + (b * x[i]) + h1;
    }

    double *Y = new double[p * q];

    for (int colnr = 0; colnr <= nelem_x; colnr = colnr + 1) {

      double c = (h[colnr] / (double)nelem_y);
      double *col = new double[q];

      for (int j = 0; j <= nelem_y; j = j + 1) {

        col[j] = (-h[colnr] / 2) + c * j;
      }
      for (int i = 0; i <= nelem_y; i = i + 1) {

        Y[i * p + colnr] = col[i];
      }
    }

    double *Coord = new double[2 * nnode];

    zeros(Coord, nnode, 2);

    for (int colnr = 0; colnr <= nelem_x; colnr = colnr + 1) {
      int c = colnr * (nelem_y + 1);
      int f = (colnr + 1) * (nelem_y + 1);

      for (int i = c; i < f; i = i + 1) {
        Coord[i * 2 + 0] = x[colnr];
      }
      for (int n = c; n < f; n = n + 1) {
        int cc = n - c;
        Coord[n * 2 + 1] = Y[cc * p + colnr];
      }
    }

    // I made the choice to keep the entiere Coord matrix to avoid any issues
    // but we obviously split our domain in two in the following parts

    // This time we compute the K matrix in each case to parallelise it

    //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //---------------------------------------------CASE
    //1--------------------------------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    if (case_num == 1) {

      //-----------------------------------------------------------Initialise
      //MPI---------------------------------------------------------------------

      int rank;
      int size;

      int err = MPI_Init(&argc, &argv);
      if (err != MPI_SUCCESS) {
        cout << "Failed to initialise MPI" << endl;
        return -1;
      }

      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      MPI_Comm_size(MPI_COMM_WORLD, &size);

      //----------------------------------------------------------------------------------------------------------------------------------------
      //-------------------------------------------------------------RANK
      //0---------------------------------------------------------------------
      //----------------------------------------------------------------------------------------------------------------------------------------

      if (rank == 0) {

        // We adapt the program for 2 cases: when nelem_x is odd or even

        int nDof2;

        int nelem_x1;

        if (nelem_x % 2 == 0) {

          nelem_x1 = (nelem_x + 2) / 2;
        }

        else {

          nelem_x1 = (nelem_x + 1) / 2;
        }

        //----- Calculation of topology matrix NodeTopo
        //-------------------------------------

        // Then we start the usual calculation of K but only for the first half
        // of the points

        // NodeTopo is therefore cut in half

        int *NodeTopo = new int[nelem_x1 * q];

        zeros_int(NodeTopo, q, nelem_x1);

        for (int colnr = 0; colnr < nelem_x1; colnr = colnr + 1) {
          for (int i = 0; i <= nelem_y; i = i + 1) {

            NodeTopo[i * nelem_x1 + colnr] = (colnr) * (nelem_y + 1) + i;
          }
        }

        //----- Calculation of topology matrix ElemNode
        //------------------------------

        // Same thing for ElemNode which will contain only half of the points

        int tai = (nelem_x1 - 1) * nelem_y;

        int *ElemNode = new int[5 * tai]; // Element connectivity

        zeros_int(ElemNode, tai, 5);

        int elemnr = 0;

        for (int colnr = 0; colnr < nelem_x1 - 1; colnr = colnr + 1) {
          for (int rownr = 0; rownr < nelem_y; rownr = rownr + 1) {

            ElemNode[elemnr * 5 + 0] = elemnr;
            ElemNode[elemnr * 5 + 4] =
                NodeTopo[(rownr + 1) * nelem_x1 + colnr]; // Lower left node
            ElemNode[elemnr * 5 + 3] =
                NodeTopo[(rownr + 1) * nelem_x1 +
                         (colnr + 1)]; // Lower right node
            ElemNode[elemnr * 5 + 2] =
                NodeTopo[rownr * nelem_x1 + (colnr + 1)]; // Upper right node
            ElemNode[elemnr * 5 + 1] =
                NodeTopo[rownr * nelem_x1 + colnr]; // upper left node
            elemnr = elemnr + 1;
          }
        }

        double *ElemX = new double[nnode_elem * tai];
        double *ElemY = new double[nnode_elem * tai];

        zeros(ElemX, tai, nnode_elem);
        zeros(ElemY, tai, nnode_elem);

        double *eNodes = new double[4];

        zeros_vect(eNodes, 4);

        double *eCoord = new double[2 * 4];

        zeros(eCoord, 4, 2);

        for (int i = 0; i < tai; i = i + 1) {
          for (int g = 1; g < 5; g = g + 1) {
            int gg = g - 1;
            eNodes[gg] = ElemNode[i * 5 + g]; // Element nodes
          }

          for (int j = 0; j < nnode_elem; j = j + 1) {
            for (int v = 0; v <= 1; v = v + 1) {

              int m = eNodes[j];
              eCoord[j * 2 + v] = Coord[m * 2 + v];
            }
          }

          for (int h = 0; h < nnode_elem; h = h + 1) {
            ElemX[i * nnode_elem + h] = eCoord[h * 2 + 0];
            ElemY[i * nnode_elem + h] = eCoord[h * 2 + 1];
          }
        }

        //----- --------------------Generate global dof numbers
        //--------------------------------

        int nnode1 = (nelem_x1) * (nelem_y + 1);

        int *globDof = new int[2 * nnode1]; // nDof/node, Dof number

        zeros_int(globDof, nnode1, 2);

        int *Top = new int[4 * tai];

        int nNode;
        int *globNodes = new int[4 * tai];

        for (int j = 0; j < tai; j = j + 1) {

          for (int s = 1; s < 5; s = s + 1) {

            int ss = s - 1;
            globNodes[j * 4 + ss] =
                ElemNode[j * 5 + s]; // Global node numbers of element nodes

            for (int k = 0; k < nnode_elem; k = k + 1) {
              nNode = ElemNode[j * 5 + (k + 1)];

              // if the already existing ndof of the present node is less than
              // the present elements ndof then replace the ndof for that node

              if (globDof[nNode * 2 + 0] < nNodeDof[k]) {
                globDof[nNode * 2 + 0] = nNodeDof[k];
              }
            }
          }
        }

        // counting the global dofs and inserting in globDof
        int nDof = 0;
        int eDof;
        for (int j = 0; j < nnode1; j = j + 1) {

          eDof = globDof[j * 2 + 0];

          for (int k = 0; k < eDof; k = k + 1) {
            globDof[j * 2 + (k + 1)] = nDof;
            nDof = nDof + 1;
          }
        }

        // First transfert between the 2 processes: "nDof2" is the total size of
        // the matrix K. I need it to define K with its full size to make
        // calculations easier
        // nonetheless K is split in two parts and for each process half of K is
        // full of 0

        MPI_Recv(&nDof2, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        //---------------------------------------------Assembly of global
        //stiffness matrix K ------------------------------
        //---------------------------------------------------- Gauss-points and
        //weights -----------------------------------

        int gauss = gaussorder; // Gauss order

        // Points
        double *GP = new double[2];

        GP[0] = -1 / pow(3, 0.5);
        GP[1] = 1 / pow(3, 0.5);

        // Weights

        int *W = new int[2];

        W[0] = 1;
        W[1] = 1;

        //----- Conductivity matrix D  -----------

        double *D = new double[2 * 2];

        D[0 * 2 + 0] = kxx;
        D[0 * 2 + 1] = kxy;
        D[1 * 2 + 0] = kxy;
        D[1 * 2 + 1] = kyy;

        //----------------------------------------

        double *K = new double[nDof2 * nDof2]; // Initiation of global stiffness
                                               // matrix K, full size

        zeros(K, nDof2, nDof2);

        for (int i = 0; i < tai; i = i + 1) {

          // - data for element i

          for (int g = 1; g < 5; g = g + 1) {
            int gg = g - 1;
            eNodes[gg] = ElemNode[i * 5 + g]; // Element nodes
          }

          for (int j = 0; j < nnode_elem; j = j + 1) {
            for (int v = 0; v <= 1; v = v + 1) {
              int m = eNodes[j];
              eCoord[j * 2 + v] = Coord[m * 2 + v];
            }
          }

          int *gDof = new int[nnode_elem]; // used to constuct scatter matrix
          int gDofNode;
          for (int j = 0; j < nnode_elem; j = j + 1) {

            // global dof for node j
            int m = eNodes[j];
            int NoDf = nNodeDof[j] + 1;

            for (int k = 1; k < NoDf; k = k + 1) {
              int kk = k - 1;
              gDofNode = globDof[m * 2 + k];
            }

            gDof[j] = gDofNode;
          }

          //- Local stiffnessmatrix, Ke, is found
          //----- Element stiffness matrix, Ke, by Gauss integration -----------

          double *Ke = new double[nnode_elem * nnode_elem];

          zeros(Ke, nnode_elem, nnode_elem);

          double *DetJ =
              new double[gauss * gauss]; // For storing the determinants of J

          double *XX = new double[nnode_elem];
          double *YY = new double[nnode_elem];

          for (int h = 0; h < 4; h = h + 1) {
            XX[h] = eCoord[h * 2 + 0];
            YY[h] = eCoord[h * 2 + 1];
          }

          for (int ii = 0; ii < gauss; ii = ii + 1) {
            for (int jj = 0; jj < gauss; jj = jj + 1) {

              float eta = GP[ii];
              float xi = GP[jj];
              // shape functions matrix
              double *N = new double[4];
              N[0] = 0.25 * (1 - xi) * (1 - eta);
              N[1] = 0.25 * (1 + xi) * (1 - eta);
              N[2] = 0.25 * (1 + xi) * (1 + eta);
              N[3] = 0.25 * (1 - xi) * (1 + eta);
              // derivative (gradient) of the shape functions

              double *GN = new double[4 * 2];
              GN[0 * 4 + 0] = -0.25 * (1 - eta);
              GN[0 * 4 + 1] = 0.25 * (1 - eta);
              GN[0 * 4 + 2] = (1 + eta) * 0.25;
              GN[0 * 4 + 3] = -(1 + eta) * 0.25;

              GN[1 * 4 + 0] = -0.25 * (1 - xi);
              GN[1 * 4 + 1] = -0.25 * (1 + xi);
              GN[1 * 4 + 2] = 0.25 * (1 + xi);
              GN[1 * 4 + 3] = 0.25 * (1 - xi);

              double *J = new double[2 * 2];

              multiply_mat_man(GN, eCoord, J, 2, 2, 4);

              double DetJ;

              DetJ = J[0 * 2 + 0] * J[1 * 2 + 1] - J[0 * 2 + 1] * J[1 * 2 + 0];

              inverse(J, 2);

              double *B = new double[4 * 2];

              multiply_mat_man(J, GN, B, 2, 4, 2);

              double *Bt = new double[2 * 4];

              transpose(B, 2, 4, Bt);

              double *dot = new double[2 * 4];

              multiply_mat_man(Bt, D, dot, 4, 2, 2);

              double *ddot = new double[4 * 4];

              multiply_mat_man(dot, B, ddot, 4, 4, 2);

              for (int o = 0; o < nnode_elem; o = o + 1) {
                for (int x = 0; x < nnode_elem; x = x + 1) {
                  Ke[o * nnode_elem + x] =
                      Ke[o * nnode_elem + x] +
                      ddot[o * 4 + x] * tp * DetJ * W[ii] * W[jj];
                }
              }
            }
          }

          for (int v = 0; v < nnode_elem; v = v + 1) {
            int b = gDof[v];

            for (int d = 0; d < nnode_elem; d = d + 1) {

              int c = gDof[d];

              K[b * nDof2 + c] = K[b * nDof2 + c] + Ke[v * nnode_elem + d];
            }
          }
        }

        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------

        nDof = nDof2;

        //----- Apply boundary conditions ----------------- Essential
        //B.C.-------------------

        int *TempNodes = new int[q]; // Nodes at the left edge of the beam

        for (int v = 0; v < q; v = v + 1) {

          TempNodes[v] = NodeTopo[v * nelem_x1 + 0];
        }

        //------------------------------------------------

        int nTempNodes = q; // Number of nodes with temp BC

        double *BC =
            new double[2 *
                       nTempNodes]; // initialize the nodal temperature vector

        zeros(BC, nTempNodes, 2);

        int T0 = 10; // Temperature at boundary

        for (int v = 0; v < nTempNodes; v = v + 1) {
          BC[v * 2 + 0] = TempNodes[v];
          BC[v * 2 + 1] = T0;
        }

        //---------------- Assembling global "Force" vector ------------

        double *OrgDof = new double[nDof]; //  Original DoF number

        zeros_vect(OrgDof, nDof);

        double *T = new double[nDof]; // initialize nodal temperature vector

        int rDof = nDof; // Reduced number of DOF

        int ind[nTempNodes];

        for (int v = 0; v < nTempNodes; v = v + 1) {
          ind[v] = BC[v * 2 + 0];
        }

        for (int v = 0; v < nTempNodes; v = v + 1) {
          int t = ind[v];
          OrgDof[t] = -1;
        }

        for (int v = 0; v < nTempNodes; v = v + 1) {
          int t = ind[v];
          OrgDof[t] = -1;
        }

        for (int v = 0; v < nTempNodes; v = v + 1) {
          int t = ind[v];
          T[t] = BC[v * 2 + 1];
        }

        rDof = rDof - nTempNodes;

        int RedDof[rDof];
        int counter1 = 0;

        for (int v = 0; v < nDof; v = v + 1) {
          if (OrgDof[v] == 0) {
            OrgDof[v] = counter1;
            RedDof[counter1] = v;
            counter1 = counter1 + 1;
          }
        }

        // Partition matrices

        int *mask_E = new int[nDof];

        for (int v = 0; v < nDof; v = v + 1) {
          for (int b = 0; b < nTempNodes; b = b + 1) {
            float bb = TempNodes[b];
            if (v == bb) {
              mask_E[v] = 1;
              break;
            } else {
              mask_E[v] = 0;
            }
          }
        }

        //-------------------------------------------------------------------------------------------------------------------------------------------------

        int mask_Ec = 0;

        for (int v = 0; v < nDof; v = v + 1) {
          mask_Ec = mask_Ec + mask_E[v];
        }
        //-----------------------------------------------

        int *mask_EE = new int[mask_Ec];

        int co = 0;

        for (int v = 0; v < nDof; v = v + 1) {

          if (mask_E[v] == 1) {

            mask_EE[co] = v;
            co = co + 1;
          }
        }

        //------------------------------------

        double *T_E = new double[mask_Ec];

        for (int v = 0; v < mask_Ec; v = v + 1) {
          int bb = mask_EE[v];
          T_E[v] = T[bb];
        }

        //-------------------------------------------------

        int fmask_Ec = 0;

        for (int v = 0; v < nDof; v = v + 1) {
          if (mask_E[v] == 0) {
            fmask_Ec = fmask_Ec + 1;
          }
        }

        //------------------------------------

        int *fmask_EE = new int[fmask_Ec];

        int fco = 0;

        for (int v = 0; v < nDof; v = v + 1) {

          if (mask_E[v] == 0) {

            fmask_EE[fco] = v;
            fco = fco + 1;
          }
        }

        MPI_Send(&fmask_Ec, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

        MPI_Send(fmask_EE, fmask_Ec, MPI_INT, 1, 0, MPI_COMM_WORLD);

        // TEST

        int nDof_total = nDof2 * nDof2;

        double *K2 = new double[nDof2 * nDof2];

        MPI_Recv(K2, nDof_total, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        for (int v = 0; v < nDof2; v = v + 1) {

          for (int b = 0; b < nDof2; b = b + 1) {

            K[v * nDof2 + b] = K[v * nDof2 + b] + K2[v * nDof2 + b];
          }
        }

        //--------------------------------------------------------------------------------------------------------

        double *K_EE = new double[mask_Ec * mask_Ec];

        zeros(K_EE, mask_Ec, mask_Ec);

        for (int v = 0; v < mask_Ec; v = v + 1) {
          int b = mask_EE[v];

          for (int d = 0; d < mask_Ec; d = d + 1) {

            int c = mask_EE[d];
            K_EE[v * mask_Ec + d] = K[b * nDof + c];
          }
        }

        //---------------------------------------------------------------------------------------------------------------------------

        double *K_FF = new double[fmask_Ec * fmask_Ec];

        zeros(K_FF, fmask_Ec, fmask_Ec);

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          int b = fmask_EE[v];

          for (int d = 0; d < fmask_Ec; d = d + 1) {

            int c = fmask_EE[d];
            K_FF[v * fmask_Ec + d] = K[b * nDof + c];
          }
        }

        //---------------------------------------------------------------------------------------------

        double *K_EF = new double[fmask_Ec * mask_Ec];

        zeros(K_EF, mask_Ec, fmask_Ec);

        for (int v = 0; v < mask_Ec; v = v + 1) {
          int b = mask_EE[v];

          for (int d = 0; d < fmask_Ec; d = d + 1) {

            int c = fmask_EE[d];

            K_EF[v * fmask_Ec + d] = K[b * nDof + c];
          }
        }

        // solve for d_F

        double *rhs = new double[fmask_Ec];

        double *K_EFt = new double[mask_Ec * fmask_Ec];

        transpose(K_EF, mask_Ec, fmask_Ec, K_EFt);

        double *prod = new double[fmask_Ec];

        for (int o = 0; o < fmask_Ec; o = o + 1) {

          prod[o] = 0;
          for (int k = 0; k < mask_Ec; k = k + 1) {
            prod[o] = prod[o] + K_EFt[o * mask_Ec + k] * T_E[k];
          }
        }

        MPI_Send(prod, fmask_Ec, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);

        double *f_F = new double[fmask_Ec];

        MPI_Recv(f_F, fmask_Ec, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          rhs[v] = f_F[v] - prod[v];
        }

        double *A = new double[fmask_Ec * fmask_Ec];

        for (int b = 0; b < fmask_Ec; b = b + 1) {
          for (int v = 0; v < fmask_Ec; v = v + 1) {
            A[b * fmask_Ec + v] = K_FF[b * fmask_Ec + v];
          }
        }

        double *T_F = new double[fmask_Ec];

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          T_F[v] = rhs[v];
        }

        //-----------------

        int *ipiv = new int[fmask_Ec];

        int info = 0;

        F77NAME(dgesv)(fmask_Ec, 1, A, fmask_Ec, ipiv, T_F, fmask_Ec, info);

        // reconstruct the global displacement d

        for (int v = 0; v < mask_Ec; v = v + 1) {
          int bb = mask_EE[v];
          T[bb] = T_E[v];
        }

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          int bb = fmask_EE[v];
          T[bb] = T_F[v];
        }

        for (int v = 0; v < mask_Ec; v = v + 1) {
          int bb = mask_EE[v];
          T[bb] = T_E[v];
        }

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          int bb = fmask_EE[v];
          T[bb] = T_F[v];
        }
      }

      //----------------------------------------------------------------------------------------------------------------------------------------
      //-------------------------------------------------------------RANK
      //1---------------------------------------------------------------------
      //----------------------------------------------------------------------------------------------------------------------------------------

      if (rank == 1) {

        // As said before, we separate the odd and even cases to split the
        // points

        int nelem_x1;

        if (nelem_x % 2 == 0) {

          nelem_x1 = (nelem_x + 2) / 2;
        }

        else {

          nelem_x1 = (nelem_x + 1) / 2;
        }

        int nelem_x2;

        if (nelem_x % 2 == 0) {

          nelem_x2 = (nelem_x + 2) / 2;
        }

        else {

          nelem_x2 = (nelem_x + 1) / 2;
        }

        int diffnelem_x2;

        diffnelem_x2 = (nelem_x) + 1 - nelem_x2 + 1;

        //----- Calculation of topology matrix NodeTopo
        //-------------------------------------

        int *NodeTopo = new int[diffnelem_x2 * q];

        zeros_int(NodeTopo, q, diffnelem_x2);

        for (int colnr = nelem_x2 - 1; colnr < nelem_x + 1; colnr = colnr + 1) {
          for (int i = 0; i <= nelem_y; i = i + 1) {
            int colnrr = colnr - nelem_x2 + 1;

            NodeTopo[i * diffnelem_x2 + colnrr] = (colnr) * (nelem_y + 1) + i;
          }
        }

        // We include the element vertices along the partition boundary in
        // NodeTopo to take them into account so all the following calculations
        // will use this column

        //----- Calculation of topology matrix ElemNode
        //------------------------------

        int tai = (nelem_x1 - 1) * nelem_y;

        int taille = nelem - tai;

        int *ElemNode = new int[5 * taille]; // Element connectivity

        zeros_int(ElemNode, (nelem - tai), 5);

        int elemnr = (nelem_x1 - 1) * (nelem_y);

        for (int colnr = 0; colnr < diffnelem_x2 - 1; colnr = colnr + 1) {
          for (int rownr = 0; rownr < nelem_y; rownr = rownr + 1) {

            int elemnrr = elemnr - (nelem_x1 - 1) * (nelem_y);

            ElemNode[elemnrr * 5 + 0] = elemnr;
            ElemNode[elemnrr * 5 + 4] =
                NodeTopo[(rownr + 1) * diffnelem_x2 + colnr]; // Lower left node
            ElemNode[elemnrr * 5 + 3] =
                NodeTopo[(rownr + 1) * diffnelem_x2 +
                         (colnr + 1)]; // Lower right node
            ElemNode[elemnrr * 5 + 2] =
                NodeTopo[rownr * diffnelem_x2 +
                         (colnr + 1)]; // Upper right node
            ElemNode[elemnrr * 5 + 1] =
                NodeTopo[rownr * diffnelem_x2 + colnr]; // upper left node
            elemnr = elemnr + 1;
          }
        }

        double *ElemX = new double[nnode_elem * (nelem - tai)];
        double *ElemY = new double[nnode_elem * (nelem - tai)];

        zeros(ElemX, (nelem - tai), nnode_elem);
        zeros(ElemY, (nelem - tai), nnode_elem);

        double *eNodes = new double[4];

        zeros_vect(eNodes, 4);

        double *eCoord = new double[2 * 4];

        zeros(eCoord, 4, 2);

        for (int i = 0; i < (nelem - tai); i = i + 1) {
          for (int g = 1; g < 5; g = g + 1) {
            int gg = g - 1;
            eNodes[gg] = ElemNode[i * 5 + g]; // Element nodes
          }

          for (int j = 0; j < nnode_elem; j = j + 1) {
            for (int v = 0; v <= 1; v = v + 1) {

              int m = eNodes[j];
              eCoord[j * 2 + v] = Coord[m * 2 + v];
            }
          }

          for (int h = 0; h < nnode_elem; h = h + 1) {
            ElemX[i * nnode_elem + h] = eCoord[h * 2 + 0];
            ElemY[i * nnode_elem + h] = eCoord[h * 2 + 1];
          }
        }

        //----- --------------------Generate global dof numbers
        //--------------------------------

        int nnode1 = (nelem_x1) * (nelem_y + 1);

        int nnode2 = nnode - nnode1;

        int nnode_min = ElemNode[0 * 5 + 1];

        for (int v = 2; v < 4; v = v + 1) {
          if (nnode_min > ElemNode[0 * 5 + v + 1]) {
            nnode_min = ElemNode[0 * 5 + v + 1];
          }
        }

        int nnode_min_tai = nnode - nnode_min;

        int *globDof = new int[2 * (nnode_min_tai)]; // nDof/node, Dof number

        zeros_int(globDof, nnode_min_tai, 2);

        int nNode;

        for (int j = 0; j < taille; j = j + 1) {

          for (int k = 0; k < nnode_elem; k = k + 1) {
            nNode = ElemNode[j * 5 + (k + 1)];
            nNode = nNode - nnode_min;

            if (nNode >= 0) {
              if (globDof[nNode * 2 + 0] < nNodeDof[k]) {
                globDof[nNode * 2 + 0] = nNodeDof[k];
              }
            }
          }
        }

        // counting the global dofs and inserting in globDof

        // We complete the second part of globDof

        int nDof = nnode_min;
        int nDof2 = nDof;

        int eDof;
        for (int j = 0; j < nnode_min_tai; j = j + 1) {

          eDof = globDof[j * 2 + 0];

          for (int k = 0; k < eDof; k = k + 1) {
            globDof[j * 2 + (k + 1)] = nDof2;
            nDof2 = nDof2 + 1;
          }
        }

        // nDof2 is now the size of T and the matrix K and we send it to the
        // process rank 0

        MPI_Send(&nDof2, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);

        //---------------------------------------------Assembly of global
        //stiffness matrix K ------------------------------
        //---------------------------------------------------- Gauss-points and
        //weights -----------------------------------

        int gauss = gaussorder; // Gauss order

        // Points
        double *GP = new double[2];

        GP[0] = -1 / pow(3, 0.5);
        GP[1] = 1 / pow(3, 0.5);

        // Weights

        int *W = new int[2];

        W[0] = 1;
        W[1] = 1;

        //----- Conductivity matrix D  -----------

        double *D = new double[2 * 2];

        D[0 * 2 + 0] = kxx;
        D[0 * 2 + 1] = kxy;
        D[1 * 2 + 0] = kxy;
        D[1 * 2 + 1] = kyy;

        //----------------------------------------

        double *K =
            new double[nDof2 *
                       nDof2]; // Initiation of global stiffness matrix K

        zeros(K, nDof2, nDof2);

        for (int i = 0; i < taille; i = i + 1) {

          // - data for element i

          for (int g = 1; g < 5; g = g + 1) {
            int gg = g - 1;
            eNodes[gg] = ElemNode[i * 5 + g]; // Element nodes
          }

          for (int j = 0; j < nnode_elem; j = j + 1) {
            for (int v = 0; v <= 1; v = v + 1) {
              int m = eNodes[j];
              eCoord[j * 2 + v] = Coord[m * 2 + v];
            }
          }

          int *gDof = new int[nnode_elem]; // used to constuct scatter matrix
          int gDofNode;

          for (int j = 0; j < nnode_elem; j = j + 1) {

            // global dof for node j

            int m = eNodes[j] - nnode_min;

            int NoDf = nNodeDof[j] + 1;

            for (int k = 1; k < NoDf; k = k + 1) {
              int kk = k - 1;

              gDofNode = globDof[m * 2 + k];
            }

            gDof[j] = gDofNode;
          }

          //- Local stiffnessmatrix, Ke, is found
          //----- Element stiffness matrix, Ke, by Gauss integration -----------

          double *Ke = new double[nnode_elem * nnode_elem];

          zeros(Ke, nnode_elem, nnode_elem);

          double *DetJ =
              new double[gauss * gauss]; // For storing the determinants of J

          double *XX = new double[nnode_elem];
          double *YY = new double[nnode_elem];

          for (int h = 0; h < 4; h = h + 1) {
            XX[h] = eCoord[h * 2 + 0];
            YY[h] = eCoord[h * 2 + 1];
          }

          for (int ii = 0; ii < gauss; ii = ii + 1) {
            for (int jj = 0; jj < gauss; jj = jj + 1) {

              double eta = GP[ii];
              double xi = GP[jj];

              // shape functions matrix

              double *N = new double[4];

              N[0] = 0.25 * (1 - xi) * (1 - eta);
              N[1] = 0.25 * (1 + xi) * (1 - eta);
              N[2] = 0.25 * (1 + xi) * (1 + eta);
              N[3] = 0.25 * (1 - xi) * (1 + eta);

              // derivative (gradient) of the shape functions

              double *GN = new double[4 * 2];

              GN[0 * 4 + 0] = -0.25 * (1 - eta);
              GN[0 * 4 + 1] = 0.25 * (1 - eta);
              GN[0 * 4 + 2] = (1 + eta) * 0.25;
              GN[0 * 4 + 3] = -(1 + eta) * 0.25;

              GN[1 * 4 + 0] = -0.25 * (1 - xi);
              GN[1 * 4 + 1] = -0.25 * (1 + xi);
              GN[1 * 4 + 2] = 0.25 * (1 + xi);
              GN[1 * 4 + 3] = 0.25 * (1 - xi);

              double *J = new double[2 * 2];

              multiply_mat_man(GN, eCoord, J, 2, 2, 4);

              double DetJ;

              DetJ = J[0 * 2 + 0] * J[1 * 2 + 1] - J[0 * 2 + 1] * J[1 * 2 + 0];

              // inverse(J, 2); For no apparent reason, the "inverse" function
              // is a problem here according to "valgrind". Thus, we do that
              // manually.

              for (int n = 0; n < 2; n = n + 1) {
                J[n * 2 + n] = J[n * 2 + n] / DetJ;
              }

              double *B = new double[4 * 2];

              multiply_mat_man(J, GN, B, 2, 4, 2);

              double *Bt = new double[2 * 4];

              transpose(B, 2, 4, Bt);

              double *dot = new double[2 * 4];

              multiply_mat_man(Bt, D, dot, 4, 2, 2);

              double *ddot = new double[4 * 4];

              multiply_mat_man(dot, B, ddot, 4, 4, 2);

              for (int o = 0; o < nnode_elem; o = o + 1) {
                for (int x = 0; x < nnode_elem; x = x + 1) {
                  Ke[o * nnode_elem + x] =
                      Ke[o * nnode_elem + x] +
                      ddot[o * 4 + x] * tp * DetJ * W[ii] * W[jj];
                }
              }
            }
          }

          for (int v = 0; v < nnode_elem; v = v + 1) {
            int b = gDof[v];

            for (int d = 0; d < nnode_elem; d = d + 1) {

              int c = gDof[d];

              K[b * nDof2 + c] = K[b * nDof2 + c] + Ke[v * nnode_elem + d];
            }
          }
        }

        // TEST

        double *K2 = new double[nDof2 * nDof2];

        for (int v = 0; v < nDof2; v = v + 1) {

          for (int b = 0; b < nDof2; b = b + 1) {

            K2[v * nDof2 + b] = K[v * nDof2 + b];
          }
        }

        int nDof_total = nDof2 * nDof2;

        MPI_Send(K2, nDof_total, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

        // We now have the matrix K split in two parts
        // To get the full K, we would have to sum the two matrices obtained in
        // each process
        // Nonetheless, as these two matrices share some common points, we can
        // consider that the values for the points at the boundary are correct
        // in each matrix K

        //----------------------------------------------------------------------------------------
        //----------------------------------------------------------------------------------------
        //----------------------------------------------------------------------------------------

        //----------------------------------------------------------------------------FLUX------------------------------------------------

        nDof = nDof2;

        // Compute nodal boundary flux vector --- natural B.C
        // Defined on edges

        int *fluxNodes = new int[q];

        for (int v = 0; v < q; v = v + 1) {
          fluxNodes[v] = NodeTopo[v * diffnelem_x2 + (diffnelem_x2 - 1)];
        }

        int nFluxNodes = q;

        //----- Defining load ----------------------------
        int qflux = 2500; // Constant flux at right edge of the beam

        double *n_bc = new double[(nFluxNodes - 1) * 4];

        for (int v = 0; v < q - 1; v = v + 1) {
          n_bc[0 * (nFluxNodes - 1) + v] = fluxNodes[v]; // node 1

          n_bc[2 * (nFluxNodes - 1) + v] = qflux; // flux value at node 1
          n_bc[3 * (nFluxNodes - 1) + v] = qflux; // flux value at node 2
        }

        for (int v = 1; v < q; v = v + 1) {
          int vv = v - 1;
          n_bc[1 * (nFluxNodes - 1) + vv] = fluxNodes[v]; // node 2
        }

        int nbe = nFluxNodes - 1; // Number of elements with flux load

        double *Coordt = new double[nnode * 2];

        transpose(Coord, nnode, 2, Coordt);

        double *xcoord = new double[nnode]; // Nodal coordinates
        double *ycoord = new double[nnode];

        for (int v = 0; v < nnode; v = v + 1) {
          xcoord[v] = Coordt[0 * nnode + v];
          ycoord[v] = Coordt[1 * nnode + v];
        }

        double *f = new double[nDof];

        for (int v = 0; v < nDof; v = v + 1) {
          f[v] = 0;
        }

        double *n_bce = new double[2];

        for (int i = 0; i < nbe; i = i + 1) {

          double *fq = new double[2];
          zeros_vect(fq, 2); // initialize the nodal source vector

          int node1 = n_bc[0 * (nFluxNodes - 1) + i]; // first node
          int node2 = n_bc[1 * (nFluxNodes - 1) + i]; // second node

          for (int v = 2; v < 4; v = v + 1) {
            int vv = v - 2;
            n_bce[vv] = n_bc[v * (nFluxNodes - 1) + i]; // flux value at an edge
          }

          double x1 = xcoord[node1]; // x coord of the first node
          double y1 = ycoord[node1]; // y coord of the first node
          double x2 = xcoord[node2]; // x coord of the first node
          double y2 = ycoord[node2]; // y coord of the second node

          double leng =
              pow(pow((x2 - x1), 2) + pow((y2 - y1), 2), 0.5); // edge length
          double detJ = leng / 2;                              // 1D Jacobian

          // integrate in xi direction (1D integration)

          for (int j = 0; j < gauss; j = j + 1) {

            double xi = GP[j]; // 1D  shape functions in parent domain
            double xii = 0.5 * (1 - xi);
            double xiii = 0.5 * (1 + xi);
            double N[2] = {xii, xiii};

            double flux = 0;

            for (int o = 0; o < 2; o = o + 1) {

              flux = flux + N[o] * n_bce[o];
            }

            fq[0] = fq[0] + W[j] * N[0] * flux * detJ * tp;
            fq[1] = fq[1] + W[j] * N[1] * flux * detJ * tp;
          }
          //  define flux as negative integrals
          fq[0] = -fq[0];
          fq[1] = -fq[1];

          f[node1] = f[node1] + fq[0];
          f[node2] = f[node2] + fq[1];
        }

        int fmask_Ec;

        MPI_Recv(&fmask_Ec, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        int *fmask_EE = new int[fmask_Ec];

        MPI_Recv(fmask_EE, fmask_Ec, MPI_INT, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        //------------------------------------------------------

        double *f_F = new double[fmask_Ec];

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          int bb = fmask_EE[v];
          f_F[v] = f[bb];
        }

        MPI_Send(f_F, fmask_Ec, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

        //------------------------------------------------------------------

        double *K_FF = new double[fmask_Ec * fmask_Ec];

        zeros(K_FF, fmask_Ec, fmask_Ec);

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          int b = fmask_EE[v];

          for (int d = 0; d < fmask_Ec; d = d + 1) {

            int c = fmask_EE[d];
            K_FF[v * fmask_Ec + d] = K[b * nDof + c];
          }
        }

        //-------------------------------------------------------

        double *prod = new double[fmask_Ec];

        MPI_Recv(prod, fmask_Ec, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        double *rhs = new double[fmask_Ec];

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          rhs[v] = f_F[v] - prod[v];
        }

        double *A = new double[fmask_Ec * fmask_Ec];

        for (int b = 0; b < fmask_Ec; b = b + 1) {
          for (int v = 0; v < fmask_Ec; v = v + 1) {
            A[b * fmask_Ec + v] = K_FF[b * fmask_Ec + v];
          }
        }

        double *T_F = new double[fmask_Ec];

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          T_F[v] = rhs[v];
        }

        int *ipiv = new int[fmask_Ec];

        int info = 0;

        F77NAME(dgesv)(fmask_Ec, 1, A, fmask_Ec, ipiv, T_F, fmask_Ec, info);

        //-------------------------------------------

        double *T = new double[nDof];

        zeros_vect(T, nDof);

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          int bb = fmask_EE[v];
          T[bb] = T_F[v];
        }
      }

      MPI_Finalize();

      return 0;
    }

    //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //---------------------------------------------CASE
    //2--------------------------------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    if (case_num == 2) {

      //-----------------------------------------------------------Initialise
      //MPI---------------------------------------------------------------------

      int rank;
      int size;

      int nDof2;

      int tail_T;

      // We define some arrays which will contain the values of T or ElemNode of
      // each process to gather our results afterwards
      // Despite my desire to automate the script, I finally defined some random
      // logical sizes due to a lack of time

      double *Ttransi = new double[50];

      zeros_vect(Ttransi, 50);

      int tail_Elem;

      int *ElemNodeTransi = new int[5 * 50];

      zeros_int(ElemNodeTransi, 50, 5);

      int err = MPI_Init(&argc, &argv);
      if (err != MPI_SUCCESS) {
        cout << "Failed to initialise MPI" << endl;
        return -1;
      }

      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      MPI_Comm_size(MPI_COMM_WORLD, &size);

      //----------------------------------------------------------------------------------------------------------------------------------------
      //-------------------------------------------------------------RANK
      //0---------------------------------------------------------------------
      //----------------------------------------------------------------------------------------------------------------------------------------

      //----- Calculation of topology matrix NodeTopo
      //-------------------------------------

      if (rank == 0) {

        int nelem_x1;

        if (nelem_x % 2 == 0) {

          nelem_x1 = (nelem_x + 2) / 2;
        }

        else {

          nelem_x1 = (nelem_x + 1) / 2;
        }

        int *NodeTopo = new int[nelem_x1 * q];

        zeros_int(NodeTopo, q, nelem_x1);

        for (int colnr = 0; colnr < nelem_x1; colnr = colnr + 1) {
          for (int i = 0; i <= nelem_y; i = i + 1) {

            NodeTopo[i * nelem_x1 + colnr] = (colnr) * (nelem_y + 1) + i;
          }
        }

        //----- Calculation of topology matrix ElemNode
        //------------------------------

        int tai = (nelem_x1 - 1) * nelem_y;

        int *ElemNode = new int[5 * tai]; // Element connectivity

        zeros_int(ElemNode, tai, 5);

        int elemnr = 0;

        for (int colnr = 0; colnr < nelem_x1 - 1; colnr = colnr + 1) {
          for (int rownr = 0; rownr < nelem_y; rownr = rownr + 1) {

            ElemNode[elemnr * 5 + 0] = elemnr;
            ElemNode[elemnr * 5 + 4] =
                NodeTopo[(rownr + 1) * nelem_x1 + colnr]; // Lower left node
            ElemNode[elemnr * 5 + 3] =
                NodeTopo[(rownr + 1) * nelem_x1 +
                         (colnr + 1)]; // Lower right node
            ElemNode[elemnr * 5 + 2] =
                NodeTopo[rownr * nelem_x1 + (colnr + 1)]; // Upper right node
            ElemNode[elemnr * 5 + 1] =
                NodeTopo[rownr * nelem_x1 + colnr]; // upper left node
            elemnr = elemnr + 1;
          }
        }

        tail_Elem = tai;

        for (int i = 0; i < tai; i = i + 1) {
          for (int j = 0; j < 5; j = j + 1) {
            ElemNodeTransi[5 * i + j] = ElemNode[5 * i + j];
          }
        }

        double *ElemX = new double[nnode_elem * tai];
        double *ElemY = new double[nnode_elem * tai];

        zeros(ElemX, tai, nnode_elem);
        zeros(ElemY, tai, nnode_elem);

        double *eNodes = new double[4];

        zeros_vect(eNodes, 4);

        double *eCoord = new double[2 * 4];

        zeros(eCoord, 4, 2);

        for (int i = 0; i < tai; i = i + 1) {
          for (int g = 1; g < 5; g = g + 1) {
            int gg = g - 1;
            eNodes[gg] = ElemNode[i * 5 + g]; // Element nodes
          }

          for (int j = 0; j < nnode_elem; j = j + 1) {
            for (int v = 0; v <= 1; v = v + 1) {

              int m = eNodes[j];
              eCoord[j * 2 + v] = Coord[m * 2 + v];
            }
          }

          for (int h = 0; h < nnode_elem; h = h + 1) {
            ElemX[i * nnode_elem + h] = eCoord[h * 2 + 0];
            ElemY[i * nnode_elem + h] = eCoord[h * 2 + 1];
          }
        }

        //----- --------------------Generate global dof numbers
        //--------------------------------

        int nnode1 = (nelem_x1) * (nelem_y + 1);

        int *globDof = new int[2 * nnode1]; // nDof/node, Dof number

        zeros_int(globDof, nnode1, 2);

        int *Top = new int[4 * tai];

        int nNode;
        int *globNodes = new int[4 * tai];

        for (int j = 0; j < tai; j = j + 1) {

          for (int s = 1; s < 5; s = s + 1) {

            int ss = s - 1;
            globNodes[j * 4 + ss] =
                ElemNode[j * 5 + s]; // Global node numbers of element nodes

            for (int k = 0; k < nnode_elem; k = k + 1) {
              nNode = ElemNode[j * 5 + (k + 1)];

              // if the already existing ndof of the present node is less than
              // the present elements ndof then replace the ndof for that node

              if (globDof[nNode * 2 + 0] < nNodeDof[k]) {
                globDof[nNode * 2 + 0] = nNodeDof[k];
              }
            }
          }
        }

        // counting the global dofs and inserting in globDof
        int nDof = 0;
        int eDof;
        for (int j = 0; j < nnode1; j = j + 1) {

          eDof = globDof[j * 2 + 0];

          for (int k = 0; k < eDof; k = k + 1) {
            globDof[j * 2 + (k + 1)] = nDof;
            nDof = nDof + 1;
          }
        }

        tail_T = nDof; // It returns tail_T=36 and then tail_T=-1

        MPI_Recv(&nDof2, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        //---------------------------------------------Assembly of global
        //stiffness matrix K ------------------------------
        //---------------------------------------------------- Gauss-points and
        //weights -----------------------------------

        int gauss = gaussorder; // Gauss order

        // Points
        double *GP = new double[2];

        GP[0] = -1 / pow(3, 0.5);
        GP[1] = 1 / pow(3, 0.5);

        // Weights

        int *W = new int[2];

        W[0] = 1;
        W[1] = 1;

        //----- Conductivity matrix D  -----------

        double *D = new double[2 * 2];

        D[0 * 2 + 0] = kxx;
        D[0 * 2 + 1] = kxy;
        D[1 * 2 + 0] = kxy;
        D[1 * 2 + 1] = kyy;

        //----------------------------------------

        double *K =
            new double[nDof2 *
                       nDof2]; // Initiation of global stiffness matrix K

        zeros(K, nDof2, nDof2);

        for (int i = 0; i < tai; i = i + 1) {

          // - data for element i

          for (int g = 1; g < 5; g = g + 1) {
            int gg = g - 1;
            eNodes[gg] = ElemNode[i * 5 + g]; // Element nodes
          }

          for (int j = 0; j < nnode_elem; j = j + 1) {
            for (int v = 0; v <= 1; v = v + 1) {
              int m = eNodes[j];
              eCoord[j * 2 + v] = Coord[m * 2 + v];
            }
          }

          int *gDof = new int[nnode_elem]; // used to constuct scatter matrix
          int gDofNode;
          for (int j = 0; j < nnode_elem; j = j + 1) {

            // global dof for node j
            int m = eNodes[j];
            int NoDf = nNodeDof[j] + 1;

            for (int k = 1; k < NoDf; k = k + 1) {
              int kk = k - 1;
              gDofNode = globDof[m * 2 + k];
            }

            gDof[j] = gDofNode;
          }

          //- Local stiffnessmatrix, Ke, is found
          //----- Element stiffness matrix, Ke, by Gauss integration -----------

          double *Ke = new double[nnode_elem * nnode_elem];

          zeros(Ke, nnode_elem, nnode_elem);

          double *DetJ =
              new double[gauss * gauss]; // For storing the determinants of J

          double *XX = new double[nnode_elem];
          double *YY = new double[nnode_elem];

          for (int h = 0; h < 4; h = h + 1) {
            XX[h] = eCoord[h * 2 + 0];
            YY[h] = eCoord[h * 2 + 1];
          }

          for (int ii = 0; ii < gauss; ii = ii + 1) {
            for (int jj = 0; jj < gauss; jj = jj + 1) {

              float eta = GP[ii];
              float xi = GP[jj];
              // shape functions matrix
              double *N = new double[4];
              N[0] = 0.25 * (1 - xi) * (1 - eta);
              N[1] = 0.25 * (1 + xi) * (1 - eta);
              N[2] = 0.25 * (1 + xi) * (1 + eta);
              N[3] = 0.25 * (1 - xi) * (1 + eta);
              // derivative (gradient) of the shape functions

              double *GN = new double[4 * 2];
              GN[0 * 4 + 0] = -0.25 * (1 - eta);
              GN[0 * 4 + 1] = 0.25 * (1 - eta);
              GN[0 * 4 + 2] = (1 + eta) * 0.25;
              GN[0 * 4 + 3] = -(1 + eta) * 0.25;

              GN[1 * 4 + 0] = -0.25 * (1 - xi);
              GN[1 * 4 + 1] = -0.25 * (1 + xi);
              GN[1 * 4 + 2] = 0.25 * (1 + xi);
              GN[1 * 4 + 3] = 0.25 * (1 - xi);

              double *J = new double[2 * 2];

              multiply_mat_man(GN, eCoord, J, 2, 2, 4);

              double DetJ;

              DetJ = J[0 * 2 + 0] * J[1 * 2 + 1] - J[0 * 2 + 1] * J[1 * 2 + 0];

              inverse(J, 2);

              double *B = new double[4 * 2];

              multiply_mat_man(J, GN, B, 2, 4, 2);

              double *Bt = new double[2 * 4];

              transpose(B, 2, 4, Bt);

              double *dot = new double[2 * 4];

              multiply_mat_man(Bt, D, dot, 4, 2, 2);

              double *ddot = new double[4 * 4];

              multiply_mat_man(dot, B, ddot, 4, 4, 2);

              for (int o = 0; o < nnode_elem; o = o + 1) {
                for (int x = 0; x < nnode_elem; x = x + 1) {
                  Ke[o * nnode_elem + x] =
                      Ke[o * nnode_elem + x] +
                      ddot[o * 4 + x] * tp * DetJ * W[ii] * W[jj];
                }
              }
            }
          }

          for (int v = 0; v < nnode_elem; v = v + 1) {
            int b = gDof[v];

            for (int d = 0; d < nnode_elem; d = d + 1) {

              int c = gDof[d];

              K[b * nDof2 + c] = K[b * nDof2 + c] + Ke[v * nnode_elem + d];
            }
          }
        }

        // I thought it could be more accurate to solve the system in two
        // distinct parts without interaction as the boundary conditions are all
        // along the x axis
        // Therefore I sent the values of the common points from a process to
        // another to have two half matrices with the good values
        // The results are available in the report

        int prems;

        MPI_Recv(&prems, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int der = globDof[2 * (nnode1 - 1) + 1];

        int tail = der - prems + 1;

        double *K2 = new double[tail * tail];

        for (int i = 0; i < tail; i = i + 1) {
          for (int j = 0; j < tail; j = j + 1) {
            K2[i * tail + j] = K[(i + prems) * nDof2 + (j + prems)];
          }
        }

        MPI_Send(&der, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

        int tailtail = tail * tail;
        MPI_Send(K2, tailtail, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);

        //--------------------------------------------------------------------------------------------------
        //--------------------------------------------------------------------------------------------------
        //--------------------------------------------------------------------------------------------------

        // Compute nodal boundary flux vector --- natural B.C
        // Defined on edges
        int *fluxNodes = new int[nelem_x1];

        for (int v = 0; v < nelem_x1; v = v + 1) {
          fluxNodes[v] = NodeTopo[(q - 1) * nelem_x1 + v];
        }

        int nFluxNodes = nelem_x1;

        //----- Defining load ----------------------------
        int qflux = 2500; // Constant flux at right edge of the beam

        double *n_bc = new double[(nFluxNodes - 1) * 4];

        for (int v = 0; v < nFluxNodes - 1; v = v + 1) {
          n_bc[0 * (nFluxNodes - 1) + v] = fluxNodes[v]; // node 1

          n_bc[2 * (nFluxNodes - 1) + v] = qflux; // flux value at node 1
          n_bc[3 * (nFluxNodes - 1) + v] = qflux; // flux value at node 2
        }

        for (int v = 1; v < nFluxNodes; v = v + 1) {
          int vv = v - 1;
          n_bc[1 * (nFluxNodes - 1) + vv] = fluxNodes[v]; // node 2
        }

        int nbe = nFluxNodes - 1; // Number of elements with flux load

        double *Coordt = new double[nnode * 2];

        transpose(Coord, nnode, 2, Coordt);

        double *xcoord = new double[nnode]; // Nodal coordinates
        double *ycoord = new double[nnode];

        for (int v = 0; v < nnode; v = v + 1) {
          xcoord[v] = Coordt[0 * nnode + v];
          ycoord[v] = Coordt[1 * nnode + v];
        }

        double *f = new double[nDof];

        for (int v = 0; v < nDof; v = v + 1) {
          f[v] = 0;
        }

        double *n_bce = new double[2];

        for (int i = 0; i < nbe; i = i + 1) {

          double *fq = new double[2];
          zeros_vect(fq, 2); // initialize the nodal source vector

          int node1 = n_bc[0 * (nFluxNodes - 1) + i]; // first node
          int node2 = n_bc[1 * (nFluxNodes - 1) + i]; // second node

          for (int v = 2; v < 4; v = v + 1) {
            int vv = v - 2;
            n_bce[vv] = n_bc[v * (nFluxNodes - 1) + i]; // flux value at an edge
          }

          double x1 = xcoord[node1]; // x coord of the first node
          double y1 = ycoord[node1]; // y coord of the first node
          double x2 = xcoord[node2]; // x coord of the first node
          double y2 = ycoord[node2]; // y coord of the second node

          double leng =
              pow(pow((x2 - x1), 2) + pow((y2 - y1), 2), 0.5); // edge length
          double detJ = leng / 2;                              // 1D Jacobian

          // integrate in xi direction (1D integration)

          for (int j = 0; j < gauss; j = j + 1) {

            double xi = GP[j]; // 1D  shape functions in parent domain
            double xii = 0.5 * (1 - xi);
            double xiii = 0.5 * (1 + xi);
            double N[2] = {xii, xiii};

            double flux = 0;

            for (int o = 0; o < 2; o = o + 1) {

              flux = flux + N[o] * n_bce[o];
            }

            fq[0] = fq[0] + W[j] * N[0] * flux * detJ * tp;
            fq[1] = fq[1] + W[j] * N[1] * flux * detJ * tp;
          }
          //  define flux as negative integrals
          fq[0] = -fq[0];
          fq[1] = -fq[1];

          f[node1] = f[node1] + fq[0];
          f[node2] = f[node2] + fq[1];
        }

        for (int i = nbe - 1; i < nbe; i = i + 1) {

          double *fq = new double[2];
          zeros_vect(fq, 2); // initialize the nodal source vector

          int node1 = n_bc[0 * (nFluxNodes - 1) + i]; // first node
          int node2 = n_bc[1 * (nFluxNodes - 1) + i]; // second node

          for (int v = 2; v < 4; v = v + 1) {
            int vv = v - 2;
            n_bce[vv] = n_bc[v * (nFluxNodes - 1) + i]; // flux value at an edge
          }

          double x1 = xcoord[node1]; // x coord of the first node
          double y1 = ycoord[node1]; // y coord of the first node
          double x2 = xcoord[node2]; // x coord of the first node
          double y2 = ycoord[node2]; // y coord of the second node

          double leng =
              pow(pow((x2 - x1), 2) + pow((y2 - y1), 2), 0.5); // edge length
          double detJ = leng / 2;                              // 1D Jacobian

          // integrate in xi direction (1D integration)

          for (int j = 0; j < gauss; j = j + 1) {

            double xi = GP[j]; // 1D  shape functions in parent domain
            double xii = 0.5 * (1 - xi);
            double xiii = 0.5 * (1 + xi);
            double N[2] = {xii, xiii};

            double flux = 0;

            for (int o = 0; o < 2; o = o + 1) {

              flux = flux + N[o] * n_bce[o];
            }

            fq[0] = fq[0] + W[j] * N[0] * flux * detJ * tp;
            fq[1] = fq[1] + W[j] * N[1] * flux * detJ * tp;
          }
          //  define flux as negative integrals
          fq[0] = -fq[0];
          fq[1] = -fq[1];

          // f[node1] = f[node1]+fq[0];
          f[node2] = f[node2] + fq[1];
        }

        //----- Apply boundary conditions ----------------- Essential
        //B.C.-------------------

        int *TempNodes =
            new int[nelem_x1]; // Nodes at the left edge of the beam

        for (int v = 0; v < nelem_x1; v = v + 1) {

          TempNodes[v] = NodeTopo[0 * nelem_x1 + v];
        }

        //------------------------------------------------

        int nTempNodes = nelem_x1; // Number of nodes with temp BC

        double *BC =
            new double[2 *
                       nTempNodes]; // initialize the nodal temperature vector

        zeros(BC, nTempNodes, 2);

        int T0 = 10; // Temperature at boundary

        for (int v = 0; v < nTempNodes; v = v + 1) {
          BC[v * 2 + 0] = TempNodes[v];
          BC[v * 2 + 1] = T0;
        }

        //---------------- Assembling global "Force" vector ------------

        double *OrgDof = new double[nDof]; //  Original DoF number

        zeros_vect(OrgDof, nDof);

        double *T = new double[nDof]; // initialize nodal temperature vector

        int rDof = nDof; // Reduced number of DOF

        int ind[nTempNodes];

        for (int v = 0; v < nTempNodes; v = v + 1) {
          ind[v] = BC[v * 2 + 0];
        }

        for (int v = 0; v < nTempNodes; v = v + 1) {
          int t = ind[v];
          OrgDof[t] = -1;
        }

        for (int v = 0; v < nTempNodes; v = v + 1) {
          int t = ind[v];
          OrgDof[t] = -1;
        }

        for (int v = 0; v < nTempNodes; v = v + 1) {
          int t = ind[v];
          T[t] = BC[v * 2 + 1];
        }

        rDof = rDof - nTempNodes;

        int RedDof[rDof];
        int counter1 = 0;

        for (int v = 0; v < nDof; v = v + 1) {
          if (OrgDof[v] == 0) {
            OrgDof[v] = counter1;
            RedDof[counter1] = v;
            counter1 = counter1 + 1;
          }
        }

        // Partition matrices

        int *mask_E = new int[nDof];

        for (int v = 0; v < nDof; v = v + 1) {
          for (int b = 0; b < q; b = b + 1) {
            float bb = TempNodes[b];
            if (v == bb) {
              mask_E[v] = 1;
              break;
            } else {
              mask_E[v] = 0;
            }
          }
        }

        //-------------------------------------------------------------------------------------------------------------------------------------------------

        int mask_Ec = 0;

        for (int v = 0; v < nDof; v = v + 1) {
          mask_Ec = mask_Ec + mask_E[v];
        }

        int *mask_EE = new int[mask_Ec];

        int co = 0;

        for (int v = 0; v < nDof; v = v + 1) {

          if (mask_E[v] == 1) {

            mask_EE[co] = v;
            co = co + 1;
          }
        }

        double *T_E = new double[mask_Ec];

        for (int v = 0; v < mask_Ec; v = v + 1) {
          int bb = mask_EE[v];
          T_E[v] = T[bb];
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------

        int fmask_Ec = 0;

        for (int v = 0; v < nDof; v = v + 1) {
          if (mask_E[v] == 0) {
            fmask_Ec = fmask_Ec + 1;
          }
        }

        int *fmask_EE = new int[fmask_Ec];

        int fco = 0;

        for (int v = 0; v < nDof; v = v + 1) {

          if (mask_E[v] == 0) {

            fmask_EE[fco] = v;
            fco = fco + 1;
          }
        }

        double *f_F = new double[fmask_Ec];

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          int bb = fmask_EE[v];
          f_F[v] = f[bb];
        }

        //---------------------------------------------------------------------------------------------------------------------------

        double *K_FF = new double[fmask_Ec * fmask_Ec];

        zeros(K_FF, fmask_Ec, fmask_Ec);

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          int b = fmask_EE[v];

          for (int d = 0; d < fmask_Ec; d = d + 1) {

            int c = fmask_EE[d];
            K_FF[v * fmask_Ec + d] = K[b * nDof2 + c];
          }
        }

        //---------------------------------------------------------------------------------------------

        double *K_EF = new double[fmask_Ec * mask_Ec];

        zeros(K_EF, mask_Ec, fmask_Ec);

        for (int v = 0; v < mask_Ec; v = v + 1) {
          int b = mask_EE[v];

          for (int d = 0; d < fmask_Ec; d = d + 1) {

            int c = fmask_EE[d];

            K_EF[v * fmask_Ec + d] = K[b * nDof2 + c];
          }
        }

        // solve for d_F

        double *rhs = new double[fmask_Ec];

        double *K_EFt = new double[mask_Ec * fmask_Ec];

        transpose(K_EF, mask_Ec, fmask_Ec, K_EFt);

        double *prod = new double[fmask_Ec];

        for (int o = 0; o < fmask_Ec; o = o + 1) {

          prod[o] = 0;
          for (int k = 0; k < mask_Ec; k = k + 1) {
            prod[o] = prod[o] + K_EFt[o * mask_Ec + k] * T_E[k];
          }
        }

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          rhs[v] = f_F[v] - prod[v];
        }

        double *A = new double[fmask_Ec * fmask_Ec];

        for (int b = 0; b < fmask_Ec; b = b + 1) {
          for (int v = 0; v < fmask_Ec; v = v + 1) {
            A[b * fmask_Ec + v] = K_FF[b * fmask_Ec + v];
          }
        }

        double *T_F = new double[fmask_Ec];

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          T_F[v] = rhs[v];
        }

        int *ipiv = new int[fmask_Ec];

        int info = 0;

        F77NAME(dgesv)(fmask_Ec, 1, A, fmask_Ec, ipiv, T_F, fmask_Ec, info);

        // reconstruct the global displacement d

        for (int v = 0; v < mask_Ec; v = v + 1) {
          int bb = mask_EE[v];
          T[bb] = T_E[v];
        }

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          int bb = fmask_EE[v];
          T[bb] = T_F[v];
        }

        // Ttransi is only defined to save the array T and allows our program to
        // gather the 2 T arrays after the calculations

        for (int i = 0; i < nDof; i = i + 1) {
          Ttransi[i] = T[i];
        }
      }

      //----------------------------------------------------------------------------------------------------------------------------------------
      //-------------------------------------------------------------RANK
      //1---------------------------------------------------------------------
      //----------------------------------------------------------------------------------------------------------------------------------------

      //----- Calculation of topology matrix NodeTopo
      //-------------------------------------

      if (rank == 1) {

        int nelem_x1;

        if (nelem_x % 2 == 0) {

          nelem_x1 = (nelem_x + 2) / 2;
        }

        else {

          nelem_x1 = (nelem_x + 1) / 2;
        }

        int nelem_x2;

        if (nelem_x % 2 == 0) {

          nelem_x2 = (nelem_x + 2) / 2;
        }

        else {

          nelem_x2 = (nelem_x + 1) / 2;
        }

        int diffnelem_x2;

        diffnelem_x2 = (nelem_x) + 1 - nelem_x2 + 1;

        int *NodeTopo = new int[diffnelem_x2 * q];

        zeros_int(NodeTopo, q, diffnelem_x2);

        for (int colnr = nelem_x2 - 1; colnr < nelem_x + 1; colnr = colnr + 1) {
          for (int i = 0; i <= nelem_y; i = i + 1) {
            int colnrr = colnr - nelem_x2 + 1;

            NodeTopo[i * diffnelem_x2 + colnrr] = (colnr) * (nelem_y + 1) + i;
          }
        }

        //----- Calculation of topology matrix ElemNode
        //------------------------------

        int tai = (nelem_x1 - 1) * nelem_y;

        int taille = nelem - tai;

        int *ElemNode = new int[5 * taille]; // Element connectivity

        zeros_int(ElemNode, (nelem - tai), 5);

        int elemnr = (nelem_x1 - 1) * (nelem_y);

        for (int colnr = 0; colnr < diffnelem_x2 - 1; colnr = colnr + 1) {
          for (int rownr = 0; rownr < nelem_y; rownr = rownr + 1) {

            int elemnrr = elemnr - (nelem_x1 - 1) * (nelem_y);

            ElemNode[elemnrr * 5 + 0] = elemnr;
            ElemNode[elemnrr * 5 + 4] =
                NodeTopo[(rownr + 1) * diffnelem_x2 + colnr]; // Lower left node
            ElemNode[elemnrr * 5 + 3] =
                NodeTopo[(rownr + 1) * diffnelem_x2 +
                         (colnr + 1)]; // Lower right node
            ElemNode[elemnrr * 5 + 2] =
                NodeTopo[rownr * diffnelem_x2 +
                         (colnr + 1)]; // Upper right node
            ElemNode[elemnrr * 5 + 1] =
                NodeTopo[rownr * diffnelem_x2 + colnr]; // upper left node
            elemnr = elemnr + 1;
          }
        }

        tail_Elem = (nelem - tai);

        for (int i = 0; i < taille; i = i + 1) {
          for (int j = 0; j < 5; j = j + 1) {
            ElemNodeTransi[5 * i + j] = ElemNode[5 * i + j];
          }
        }

        double *ElemX = new double[nnode_elem * (nelem - tai)];
        double *ElemY = new double[nnode_elem * (nelem - tai)];

        zeros(ElemX, (nelem - tai), nnode_elem);
        zeros(ElemY, (nelem - tai), nnode_elem);

        double *eNodes = new double[4];

        zeros_vect(eNodes, 4);

        double *eCoord = new double[2 * 4];

        zeros(eCoord, 4, 2);

        for (int i = 0; i < (nelem - tai); i = i + 1) {
          for (int g = 1; g < 5; g = g + 1) {
            int gg = g - 1;
            eNodes[gg] = ElemNode[i * 5 + g]; // Element nodes
          }

          for (int j = 0; j < nnode_elem; j = j + 1) {
            for (int v = 0; v <= 1; v = v + 1) {

              int m = eNodes[j];
              eCoord[j * 2 + v] = Coord[m * 2 + v];
            }
          }

          for (int h = 0; h < nnode_elem; h = h + 1) {
            ElemX[i * nnode_elem + h] = eCoord[h * 2 + 0];
            ElemY[i * nnode_elem + h] = eCoord[h * 2 + 1];
          }
        }

        //----- --------------------Generate global dof numbers
        //--------------------------------

        int nnode1 = (nelem_x1) * (nelem_y + 1);

        int nnode2 = nnode - nnode1;

        int nnode_min = ElemNode[0 * 5 + 1];

        for (int v = 2; v < 4; v = v + 1) {
          if (nnode_min > ElemNode[0 * 5 + v + 1]) {
            nnode_min = ElemNode[0 * 5 + v + 1];
          }
        }

        int nnode_min_tai = nnode - nnode_min;

        int *globDof = new int[2 * (nnode_min_tai)]; // nDof/node, Dof number

        zeros_int(globDof, nnode_min_tai, 2);

        int nNode;

        for (int j = 0; j < taille; j = j + 1) {

          for (int k = 0; k < nnode_elem; k = k + 1) {
            nNode = ElemNode[j * 5 + (k + 1)];
            nNode = nNode - nnode_min;

            // if the already existing ndof of the present node is less than
            // the present elements ndof then replace the ndof for that node

            if (nNode >= 0) {
              if (globDof[nNode * 2 + 0] < nNodeDof[k]) {
                globDof[nNode * 2 + 0] = nNodeDof[k];
              }
            }
          }
        }

        // counting the global dofs and inserting in globDof

        // int nDof=MPI_Recv(&nDof, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
        // MPI_STATUS_IGNORE);

        int nDof = nnode_min;

        nDof2 = nDof;

        tail_T = nnode_min_tai;

        int eDof;
        for (int j = 0; j < nnode_min_tai; j = j + 1) {

          eDof = globDof[j * 2 + 0];

          for (int k = 0; k < eDof; k = k + 1) {
            globDof[j * 2 + (k + 1)] = nDof2;
            nDof2 = nDof2 + 1;
          }
        }

        MPI_Send(&nDof2, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);

        //---------------------------------------------Assembly of global
        //stiffness matrix K ------------------------------
        //---------------------------------------------------- Gauss-points and
        //weights -----------------------------------

        int gauss = gaussorder; // Gauss order

        // Points
        double *GP = new double[2];

        GP[0] = -1 / pow(3, 0.5);
        GP[1] = 1 / pow(3, 0.5);

        // Weights

        int *W = new int[2];

        W[0] = 1;
        W[1] = 1;

        //----- Conductivity matrix D  -----------

        double *D = new double[2 * 2];

        D[0 * 2 + 0] = kxx;
        D[0 * 2 + 1] = kxy;
        D[1 * 2 + 0] = kxy;
        D[1 * 2 + 1] = kyy;

        //----------------------------------------

        double *K =
            new double[nDof2 *
                       nDof2]; // Initiation of global stiffness matrix K

        zeros(K, nDof2, nDof2);

        for (int i = 0; i < taille; i = i + 1) {

          // - data for element i

          for (int g = 1; g < 5; g = g + 1) {
            int gg = g - 1;
            eNodes[gg] = ElemNode[i * 5 + g]; // Element nodes
          }

          for (int j = 0; j < nnode_elem; j = j + 1) {
            for (int v = 0; v <= 1; v = v + 1) {
              int m = eNodes[j];
              eCoord[j * 2 + v] = Coord[m * 2 + v];
            }
          }

          int *gDof = new int[nnode_elem]; // used to constuct scatter matrix
          int gDofNode;

          for (int j = 0; j < nnode_elem; j = j + 1) {

            // global dof for node j

            int m = eNodes[j] - nnode_min;

            int NoDf = nNodeDof[j] + 1;

            for (int k = 1; k < NoDf; k = k + 1) {
              int kk = k - 1;

              gDofNode = globDof[m * 2 + k];
            }

            gDof[j] = gDofNode;
          }

          //- Local stiffnessmatrix, Ke, is found
          //----- Element stiffness matrix, Ke, by Gauss integration -----------

          double *Ke = new double[nnode_elem * nnode_elem];

          zeros(Ke, nnode_elem, nnode_elem);

          double *DetJ =
              new double[gauss * gauss]; // For storing the determinants of J

          double *XX = new double[nnode_elem];
          double *YY = new double[nnode_elem];

          for (int h = 0; h < 4; h = h + 1) {
            XX[h] = eCoord[h * 2 + 0];
            YY[h] = eCoord[h * 2 + 1];
          }

          for (int ii = 0; ii < gauss; ii = ii + 1) {
            for (int jj = 0; jj < gauss; jj = jj + 1) {

              double eta = GP[ii];
              double xi = GP[jj];

              // shape functions matrix

              double *N = new double[4];

              N[0] = 0.25 * (1 - xi) * (1 - eta);
              N[1] = 0.25 * (1 + xi) * (1 - eta);
              N[2] = 0.25 * (1 + xi) * (1 + eta);
              N[3] = 0.25 * (1 - xi) * (1 + eta);

              // derivative (gradient) of the shape functions

              double *GN = new double[4 * 2];

              GN[0 * 4 + 0] = -0.25 * (1 - eta);
              GN[0 * 4 + 1] = 0.25 * (1 - eta);
              GN[0 * 4 + 2] = (1 + eta) * 0.25;
              GN[0 * 4 + 3] = -(1 + eta) * 0.25;

              GN[1 * 4 + 0] = -0.25 * (1 - xi);
              GN[1 * 4 + 1] = -0.25 * (1 + xi);
              GN[1 * 4 + 2] = 0.25 * (1 + xi);
              GN[1 * 4 + 3] = 0.25 * (1 - xi);

              double *J = new double[2 * 2];

              multiply_mat_man(GN, eCoord, J, 2, 2, 4);

              double DetJ;

              DetJ = J[0 * 2 + 0] * J[1 * 2 + 1] - J[0 * 2 + 1] * J[1 * 2 + 0];

              inverse(J, 2);

              double *B = new double[4 * 2];

              multiply_mat_man(J, GN, B, 2, 4, 2);

              double *Bt = new double[2 * 4];

              transpose(B, 2, 4, Bt);

              double *dot = new double[2 * 4];

              multiply_mat_man(Bt, D, dot, 4, 2, 2);

              double *ddot = new double[4 * 4];

              multiply_mat_man(dot, B, ddot, 4, 4, 2);

              for (int o = 0; o < nnode_elem; o = o + 1) {
                for (int x = 0; x < nnode_elem; x = x + 1) {
                  Ke[o * nnode_elem + x] =
                      Ke[o * nnode_elem + x] +
                      ddot[o * 4 + x] * tp * DetJ * W[ii] * W[jj];
                }
              }
            }
          }

          for (int v = 0; v < nnode_elem; v = v + 1) {
            int b = gDof[v];

            for (int d = 0; d < nnode_elem; d = d + 1) {

              int c = gDof[d];

              K[b * nDof2 + c] = K[b * nDof2 + c] + Ke[v * nnode_elem + d];
            }
          }
        }

        int prems = globDof[2 * 0 + 1];

        MPI_Send(&prems, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);

        int der;

        MPI_Recv(&der, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int tail = der - prems + 1;

        int tailtail = tail * tail;

        double *K2 = new double[tail * tail];

        MPI_Recv(K2, tailtail, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        for (int i = 0; i < tail; i = i + 1) {

          for (int j = 0; j < tail; j = j + 1) {

            K[(i + prems) * nDof2 + (j + prems)] =
                K[(i + prems) * nDof2 + (j + prems)] + K2[i * tail + j];
          }
        }

        nDof = nDof2;

        //----------------------------------------------------------------------------------------------------
        //----------------------------------------------------------------------------------------------------
        //----------------------------------------------------------------------------------------------------

        //----------------------------------------------------------------------------FLUX------------------------------------------------

        // Compute nodal boundary flux vector --- natural B.C
        // Defined on edges

        int *fluxNodes = new int[diffnelem_x2];

        for (int v = 0; v < diffnelem_x2; v = v + 1) {
          fluxNodes[v] = NodeTopo[(q - 1) * diffnelem_x2 + v];
        }

        int nFluxNodes = diffnelem_x2;

        int prem = fluxNodes[0];

        //----- Defining load ----------------------------
        int qflux = 2500; // Constant flux at right edge of the beam

        double *n_bc = new double[(nFluxNodes - 1) * 4];

        for (int v = 0; v < nFluxNodes - 1; v = v + 1) {
          n_bc[0 * (nFluxNodes - 1) + v] = fluxNodes[v]; // node 1

          n_bc[2 * (nFluxNodes - 1) + v] = qflux; // flux value at node 1
          n_bc[3 * (nFluxNodes - 1) + v] = qflux; // flux value at node 2
        }

        for (int v = 1; v < nFluxNodes; v = v + 1) {
          int vv = v - 1;
          n_bc[1 * (nFluxNodes - 1) + vv] = fluxNodes[v]; // node 2
        }

        int nbe = nFluxNodes - 1; // Number of elements with flux load

        double *Coordt = new double[nnode * 2];

        transpose(Coord, nnode, 2, Coordt);

        double *xcoord = new double[nnode]; // Nodal coordinates
        double *ycoord = new double[nnode];

        for (int v = 0; v < nnode; v = v + 1) {
          xcoord[v] = Coordt[0 * nnode + v];
          ycoord[v] = Coordt[1 * nnode + v];
        }

        double *f = new double[nnode_min_tai];

        for (int v = 0; v < nnode_min_tai; v = v + 1) {
          f[v] = 0;
        }

        double *n_bce = new double[2];

        for (int i = 0; i < nbe; i = i + 1) {

          double *fq = new double[2];
          zeros_vect(fq, 2); // initialize the nodal source vector

          int node1 = n_bc[0 * (nFluxNodes - 1) + i]; // first node
          int node2 = n_bc[1 * (nFluxNodes - 1) + i]; // second node

          for (int v = 2; v < 4; v = v + 1) {
            int vv = v - 2;
            n_bce[vv] = n_bc[v * (nFluxNodes - 1) + i]; // flux value at an edge
          }

          double x1 = xcoord[node1]; // x coord of the first node
          double y1 = ycoord[node1]; // y coord of the first node
          double x2 = xcoord[node2]; // x coord of the first node
          double y2 = ycoord[node2]; // y coord of the second node

          double leng =
              pow(pow((x2 - x1), 2) + pow((y2 - y1), 2), 0.5); // edge length
          double detJ = leng / 2;                              // 1D Jacobian

          // integrate in xi direction (1D integration)

          for (int j = 0; j < gauss; j = j + 1) {

            double xi = GP[j]; // 1D  shape functions in parent domain
            double xii = 0.5 * (1 - xi);
            double xiii = 0.5 * (1 + xi);
            double N[2] = {xii, xiii};

            double flux = 0;

            for (int o = 0; o < 2; o = o + 1) {

              flux = flux + N[o] * n_bce[o];
            }

            fq[0] = fq[0] + W[j] * N[0] * flux * detJ * tp;
            fq[1] = fq[1] + W[j] * N[1] * flux * detJ * tp;
          }
          //  define flux as negative integrals
          fq[0] = -fq[0];
          fq[1] = -fq[1];

          int diff = node2 - node1 - 1;

          int dnode1 = node1 - prem + diff;
          int dnode2 = node2 - prem + diff;

          f[dnode1] = f[dnode1] + fq[0];
          f[dnode2] = f[dnode2] + fq[1];
        }

        // This additional part is to get the good first boundary condition in f
        // (as we split our domain in 2, we can't consider the cut as a real
        // border)
        // The same has been done for the last boundary condition in f at rank=0
        for (int i = 0; i < 1; i = i + 1) {

          double *fq = new double[2];
          zeros_vect(fq, 2); // initialize the nodal source vector

          int node1 = n_bc[0 * (nFluxNodes - 1) + i]; // first node
          int node2 = n_bc[1 * (nFluxNodes - 1) + i]; // second node

          for (int v = 2; v < 4; v = v + 1) {
            int vv = v - 2;
            n_bce[vv] = n_bc[v * (nFluxNodes - 1) + i]; // flux value at an edge
          }

          double x1 = xcoord[node1]; // x coord of the first node
          double y1 = ycoord[node1]; // y coord of the first node
          double x2 = xcoord[node2]; // x coord of the first node
          double y2 = ycoord[node2]; // y coord of the second node

          double leng =
              pow(pow((x2 - x1), 2) + pow((y2 - y1), 2), 0.5); // edge length
          double detJ = leng / 2;                              // 1D Jacobian

          // integrate in xi direction (1D integration)

          for (int j = 0; j < gauss; j = j + 1) {

            double xi = GP[j]; // 1D  shape functions in parent domain
            double xii = 0.5 * (1 - xi);
            double xiii = 0.5 * (1 + xi);
            double N[2] = {xii, xiii};

            double flux = 0;

            for (int o = 0; o < 2; o = o + 1) {

              flux = flux + N[o] * n_bce[o];
            }

            fq[0] = fq[0] + W[j] * N[0] * flux * detJ * tp;
            fq[1] = fq[1] + W[j] * N[1] * flux * detJ * tp;
          }
          //  define flux as negative integrals
          fq[0] = -fq[0];
          fq[1] = -fq[1];

          int diff = node2 - node1 - 1;

          int dnode1 = node1 - prem + diff;
          int dnode2 = node2 - prem + diff;

          f[dnode1] = f[dnode1] + fq[0];
          // f[dnode2] = f[dnode2] + fq[1];
        }

        //----- Apply boundary conditions ----------------- Essential
        //B.C.-------------------

        int *TempNodes =
            new int[diffnelem_x2]; // Nodes at the left edge of the beam

        for (int v = 0; v < diffnelem_x2; v = v + 1) {

          TempNodes[v] = NodeTopo[0 * diffnelem_x2 + v];
        }

        //------------------------------------------------

        int nTempNodes = diffnelem_x2; // Number of nodes with temp BC

        double *BC =
            new double[2 *
                       nTempNodes]; // initialize the nodal temperature vector

        zeros(BC, nTempNodes, 2);

        int T0 = 10; // Temperature at boundary

        for (int v = 0; v < nTempNodes; v = v + 1) {
          BC[v * 2 + 0] = TempNodes[v];
          BC[v * 2 + 1] = T0;
        }

        //---------------- Assembling global "Force" vector ------------

        double *OrgDof = new double[nnode_min_tai]; //  Original DoF number

        zeros_vect(OrgDof, nnode_min_tai);

        double *T =
            new double[nnode_min_tai]; // initialize nodal temperature vector

        int rDof = nnode_min_tai; // Reduced number of DOF

        int ind[nTempNodes];

        int min_ind = BC[0 * 2 + 0];

        for (int v = 0; v < nTempNodes; v = v + 1) {
          ind[v] = BC[v * 2 + 0];
        }

        for (int v = 0; v < nTempNodes; v = v + 1) {
          int t = ind[v];
          int tt = t - min_ind;
          OrgDof[tt] = -1;
        }

        for (int v = 0; v < nTempNodes; v = v + 1) {
          int t = ind[v];
          int tt = t - min_ind;
          OrgDof[tt] = -1;
        }

        for (int v = 0; v < nTempNodes; v = v + 1) {
          int t = ind[v];
          int tt = t - min_ind;
          T[tt] = BC[v * 2 + 1];
        }

        rDof = rDof - nTempNodes;

        int RedDof[rDof];
        int counter1 = 0;

        for (int v = 0; v < nnode_min_tai; v = v + 1) {
          if (OrgDof[v] == 0) {
            OrgDof[v] = counter1;
            RedDof[counter1] = v;
            counter1 = counter1 + 1;
          }
        }

        // Partition matrices

        int *mask_E = new int[nnode_min_tai];

        for (int v = 0; v < nnode_min_tai; v = v + 1) {
          for (int b = 0; b < diffnelem_x2; b = b + 1) {
            float bb = TempNodes[b];
            bb = bb - min_ind;

            if (v == bb) {
              mask_E[v] = 1;
              break;
            } else {
              mask_E[v] = 0;
            }
          }
        }

        int mask_Ec = 0;

        for (int v = 0; v < nnode_min_tai; v = v + 1) {
          mask_Ec = mask_Ec + mask_E[v];
        }

        int *mask_EE = new int[mask_Ec];

        int co = 0;

        for (int v = 0; v < nnode_min_tai; v = v + 1) {

          if (mask_E[v] == 1) {

            mask_EE[co] = v;
            co = co + 1;
          }
        }

        double *T_E = new double[mask_Ec];

        for (int v = 0; v < mask_Ec; v = v + 1) {
          int bb = mask_EE[v];
          T_E[v] = T[bb];
        }

        //--------------------------------------------------------------------------------------------------------------------------------------------

        int fmask_Ec = 0;

        for (int v = 0; v < nnode_min_tai; v = v + 1) {
          if (mask_E[v] == 0) {
            fmask_Ec = fmask_Ec + 1;
          }
        }

        int *fmask_EE = new int[fmask_Ec];

        int fco = 0;

        for (int v = 0; v < nnode_min_tai; v = v + 1) {

          if (mask_E[v] == 0) {

            fmask_EE[fco] = v;
            fco = fco + 1;
          }
        }

        double *f_F = new double[fmask_Ec];

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          int bb = fmask_EE[v];
          f_F[v] = f[bb];
        }

        //---------------------------------------------------------------------------------------------

        double *K_EF = new double[fmask_Ec * mask_Ec];

        zeros(K_EF, mask_Ec, fmask_Ec);

        for (int v = 0; v < mask_Ec; v = v + 1) {
          int b = mask_EE[v] + min_ind;

          for (int d = 0; d < fmask_Ec; d = d + 1) {

            int c = fmask_EE[d] + min_ind;

            K_EF[v * fmask_Ec + d] = K[b * nDof2 + c];
          }
        }

        //---------------------------------------------------------------------------------------------------------------------------

        double *K_FF = new double[fmask_Ec * fmask_Ec];

        zeros(K_FF, fmask_Ec, fmask_Ec);

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          int b = fmask_EE[v] + min_ind;

          for (int d = 0; d < fmask_Ec; d = d + 1) {

            int c = fmask_EE[d] + min_ind;
            K_FF[v * fmask_Ec + d] = K[b * nDof2 + c];
          }
        }

        // solve for d_F

        double *rhs = new double[fmask_Ec];

        double *K_EFt = new double[mask_Ec * fmask_Ec];

        transpose(K_EF, mask_Ec, fmask_Ec, K_EFt);

        double *prod = new double[fmask_Ec];

        for (int o = 0; o < fmask_Ec; o = o + 1) {

          prod[o] = 0;
          for (int k = 0; k < mask_Ec; k = k + 1) {
            prod[o] = prod[o] + K_EFt[o * mask_Ec + k] * T_E[k];
          }
        }

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          rhs[v] = f_F[v] - prod[v];
        }

        double *A = new double[fmask_Ec * fmask_Ec];

        for (int b = 0; b < fmask_Ec; b = b + 1) {
          for (int v = 0; v < fmask_Ec; v = v + 1) {
            A[b * fmask_Ec + v] = K_FF[b * fmask_Ec + v];
          }
        }

        double *T_F = new double[fmask_Ec];

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          T_F[v] = rhs[v];
        }

        int *ipiv = new int[fmask_Ec];

        int info = 0;

        F77NAME(dgesv)(fmask_Ec, 1, A, fmask_Ec, ipiv, T_F, fmask_Ec, info);

        // reconstruct the global displacement d

        for (int v = 0; v < mask_Ec; v = v + 1) {
          int bb = mask_EE[v];
          T[bb] = T_E[v];
        }

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          int bb = fmask_EE[v];
          T[bb] = T_F[v];
        }

        int diff =
            NodeTopo[0 * diffnelem_x2 + 1] - NodeTopo[0 * diffnelem_x2 + 0];

        double *T2 = new double[nnode_min_tai - diff];

        for (int v = 0; v < nnode_min_tai - diff; v = v + 1) {

          T2[v] = T[v + diff];
        }

        for (int i = 0; i < nnode_min_tai; i = i + 1) {
          Ttransi[i] = T[i];
        }
      }

      // Gather the T arrays

      int tail_T2 = tail_T * 2;

      double *Ttotal = new double[tail_T2];

      zeros_vect(Ttotal, tail_T2);

      MPI_Gather(Ttransi, tail_T, MPI_DOUBLE, Ttotal, tail_T, MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);

      int difference = tail_T2 - nDof2;

      // Gather the ElemNodes matrices

      int tail_Elem2 = tail_Elem * 2;

      int *ElemNodeTotal = new int[5 * tail_Elem2];

      MPI_Gather(ElemNodeTransi, 5 * tail_Elem, MPI_INT, ElemNodeTotal,
                 5 * tail_Elem, MPI_INT, 0, MPI_COMM_WORLD);

      if (rank == 0) {

        // Here we obtained the same values for the end of the first T array and
        // the beginning of the second T array
        // It is logical as we have common points and we solved the common
        // values between the two K matrices

        // Thus, we truncate the second array T and juxtapose the two arrays to
        // get full T

        double *T = new double[nDof2];

        zeros_vect(T, nDof2);

        for (int i = 0; i < nDof2; i = i + 1) {

          if (i < tail_T) {
            T[i] = Ttotal[i];
          }

          else {
            T[i] = Ttotal[i + difference];
          }
        }

        //-----------------------------------------------Create
        //points-----------------------------------------------------

        double *points = new double[3 * nnode];

        for (int v = 0; v < nnode; v = v + 1) {
          for (int b = 0; b < 3; b = b + 1) {

            if (b != 2) {
              points[v * 3 + b] = Coord[v * 2 + b];

            }

            else {
              points[v * 3 + b] = 0;
            }
          }
        }

        //--------------------------------------------Create
        //cells---------------------------------------------------------

        int *cells = new int[4 * nelem];

        for (int i = 0; i < nelem; i = i + 1) {

          for (int j = 0; j < 4; j = j + 1) {

            int jj = j + 1;

            cells[i * 4 + j] = ElemNodeTotal[i * 5 + jj];
          }
        }

        //---------------Create VTK
        //file----------------------------------------------------------------

        // Create intro

        ofstream vOut("datap2.vtk", ios::out | ios::trunc);
        vOut << "# vtk DataFile Version 4.0" << endl;
        vOut << "vtk output" << endl;
        vOut << "ASCII" << endl;
        vOut << "DATASET UNSTRUCTURED_GRID" << endl;

        // Print points

        vOut << "POINTS"
             << " " << nnode << " "
             << "double" << endl;
        for (int v = 0; v < nnode; v = v + 1) {
          for (int b = 0; b < 3; b = b + 1) {
            vOut << points[v * 3 + b] << " ";
          }
        }
        vOut << endl;

        // print cells

        int total_num_cells = nelem;
        int total_num_idx = 5 * nelem;

        vOut << "CELLS"
             << " " << total_num_cells << " " << total_num_idx << endl;

        // Creation of keys_cells

        int *keys_cells = new int[5 * nelem];

        for (int i = 0; i < nelem; i = i + 1) {
          keys_cells[i * 5 + 0] = 4;
          for (int j = 1; j < 5; j = j + 1) {
            int jj = j - 1;
            keys_cells[i * 5 + j] = cells[i * 4 + jj];
          }
        }

        // print keys_cells

        for (int i = 0; i < total_num_cells; i = i + 1) {

          for (int j = 0; j < 5; j = j + 1) {

            vOut << keys_cells[i * 5 + j] << " ";
          }
          vOut << endl;
        }

        vOut << "CELL_TYPES"
             << " " << total_num_cells << endl;

        for (int i = 0; i < total_num_cells; i = i + 1) {

          vOut << 9 << endl;
        }

        // Here we don't create "point_data" as we directly use T and
        // len(T)=nDof

        // Print point_data

        int len_points = nnode;

        vOut << "POINT_DATA"
             << " " << len_points << endl;
        vOut << "FIELD FieldData"
             << " "
             << "1" << endl;
        vOut << "disp"
             << " "
             << "1"
             << " " << nDof2 << " "
             << "double" << endl;

        for (int i = 0; i < nDof2; i = i + 1) {
          vOut << T[i] << " ";
        }

        vOut.close();
      }

      MPI_Finalize();

      return 0;
    }

    //----------------------------------------------------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------CASE
    //3-----------------------------------------------------------------------------
    //----------------------------------------------------------------------------------------------------------------------------------------------

    if (case_num == 3) {

      //-----------------------------------------------------------Initialise
      //MPI---------------------------------------------------------------------

      int rank;
      int size;

      int nDof2;

      int tail_T;

      // all these parameters are used to solve the linear system and are
      // described in the corresponding part below
      int tailTF0;
      int mask_Eca0;
      int fmask_Eca0;
      int mask_Ecb0;
      int fmask_Ecb0;
      int nDof0;
      int lenn0;

      // Again, we define arrays to gather our results
      double *Ttransi = new double[100];

      zeros_vect(Ttransi, 100);

      int tail_Elem0;
      int tail_Elem1;

      int *ElemNodeTransi = new int[5 * 100];

      zeros_int(ElemNodeTransi, 100, 5);

      int err = MPI_Init(&argc, &argv);
      if (err != MPI_SUCCESS) {
        cout << "Failed to initialise MPI" << endl;
        return -1;
      }

      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      MPI_Comm_size(MPI_COMM_WORLD, &size);

      //----------------------------------------------------------------------------------------------------------------------------------------
      //-------------------------------------------------------------RANK
      //0---------------------------------------------------------------------
      //----------------------------------------------------------------------------------------------------------------------------------------

      //----- Calculation of topology matrix NodeTopo
      //-------------------------------------

      if (rank == 0) {

        int nelem_x1;

        if (nelem_x % 2 == 0) {

          nelem_x1 = (nelem_x + 2) / 2;
        }

        else {

          nelem_x1 = (nelem_x + 1) / 2;
        }

        int *NodeTopo = new int[nelem_x1 * q];

        zeros_int(NodeTopo, q, nelem_x1);

        for (int colnr = 0; colnr < nelem_x1; colnr = colnr + 1) {
          for (int i = 0; i <= nelem_y; i = i + 1) {

            NodeTopo[i * nelem_x1 + colnr] = (colnr) * (nelem_y + 1) + i;
          }
        }

        //----- Calculation of topology matrix ElemNode
        //------------------------------

        int tai = (nelem_x1 - 1) * nelem_y;

        int *ElemNode = new int[5 * tai]; // Element connectivity

        zeros_int(ElemNode, tai, 5);

        int elemnr = 0;

        for (int colnr = 0; colnr < nelem_x1 - 1; colnr = colnr + 1) {
          for (int rownr = 0; rownr < nelem_y; rownr = rownr + 1) {

            ElemNode[elemnr * 5 + 0] = elemnr;
            ElemNode[elemnr * 5 + 4] =
                NodeTopo[(rownr + 1) * nelem_x1 + colnr]; // Lower left node
            ElemNode[elemnr * 5 + 3] =
                NodeTopo[(rownr + 1) * nelem_x1 +
                         (colnr + 1)]; // Lower right node
            ElemNode[elemnr * 5 + 2] =
                NodeTopo[rownr * nelem_x1 + (colnr + 1)]; // Upper right node
            ElemNode[elemnr * 5 + 1] =
                NodeTopo[rownr * nelem_x1 + colnr]; // upper left node
            elemnr = elemnr + 1;
          }
        }

        tail_Elem0 = tai;
        MPI_Send(&tail_Elem0, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

        MPI_Recv(&tail_Elem1, 1, MPI_INT, 1, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        // tai_Elem1=tai_Elem1;

        for (int i = 0; i < tai; i = i + 1) {
          for (int j = 0; j < 5; j = j + 1) {
            ElemNodeTransi[5 * i + j] = ElemNode[5 * i + j];
          }
        }

        double *ElemX = new double[nnode_elem * tai];
        double *ElemY = new double[nnode_elem * tai];

        zeros(ElemX, tai, nnode_elem);
        zeros(ElemY, tai, nnode_elem);

        double *eNodes = new double[4];

        zeros_vect(eNodes, 4);

        double *eCoord = new double[2 * 4];

        zeros(eCoord, 4, 2);

        for (int i = 0; i < tai; i = i + 1) {
          for (int g = 1; g < 5; g = g + 1) {
            int gg = g - 1;
            eNodes[gg] = ElemNode[i * 5 + g]; // Element nodes
          }

          for (int j = 0; j < nnode_elem; j = j + 1) {
            for (int v = 0; v <= 1; v = v + 1) {

              int m = eNodes[j];
              eCoord[j * 2 + v] = Coord[m * 2 + v];
            }
          }

          for (int h = 0; h < nnode_elem; h = h + 1) {
            ElemX[i * nnode_elem + h] = eCoord[h * 2 + 0];
            ElemY[i * nnode_elem + h] = eCoord[h * 2 + 1];
          }
        }

        //----- --------------------Generate global dof numbers
        //--------------------------------

        int nnode1 = (nelem_x1) * (nelem_y + 1);

        int *globDof = new int[2 * nnode1]; // nDof/node, Dof number

        zeros_int(globDof, nnode1, 2);

        int *Top = new int[4 * tai];

        int nNode;
        int *globNodes = new int[4 * tai];

        for (int j = 0; j < tai; j = j + 1) {

          for (int s = 1; s < 5; s = s + 1) {

            int ss = s - 1;
            globNodes[j * 4 + ss] =
                ElemNode[j * 5 + s]; // Global node numbers of element nodes

            for (int k = 0; k < nnode_elem; k = k + 1) {
              nNode = ElemNode[j * 5 + (k + 1)];

              // if the already existing ndof of the present node is less than
              // the present elements ndof then replace the ndof for that node

              if (globDof[nNode * 2 + 0] < nNodeDof[k]) {
                globDof[nNode * 2 + 0] = nNodeDof[k];
              }
            }
          }
        }

        // counting the global dofs and inserting in globDof
        int nDof = 0;
        int eDof;
        for (int j = 0; j < nnode1; j = j + 1) {

          eDof = globDof[j * 2 + 0];

          for (int k = 0; k < eDof; k = k + 1) {
            globDof[j * 2 + (k + 1)] = nDof;
            nDof = nDof + 1;
          }
        }

        tail_T = nDof;

        MPI_Recv(&nDof2, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        //---------------------------------------------Assembly of global
        //stiffness matrix K ------------------------------
        //---------------------------------------------------- Gauss-points and
        //weights -----------------------------------

        int gauss = gaussorder; // Gauss order

        // Points
        double *GP = new double[2];

        GP[0] = -1 / pow(3, 0.5);
        GP[1] = 1 / pow(3, 0.5);

        // Weights

        int *W = new int[2];

        W[0] = 1;
        W[1] = 1;

        //----- Conductivity matrix D  -----------

        double *D = new double[2 * 2];

        D[0 * 2 + 0] = kxx;
        D[0 * 2 + 1] = kxy;
        D[1 * 2 + 0] = kxy;
        D[1 * 2 + 1] = kyy;

        //----------------------------------------

        double *K =
            new double[nDof2 *
                       nDof2]; // Initiation of global stiffness matrix K

        zeros(K, nDof2, nDof2);

        for (int i = 0; i < tai; i = i + 1) {

          // - data for element i

          for (int g = 1; g < 5; g = g + 1) {
            int gg = g - 1;
            eNodes[gg] = ElemNode[i * 5 + g]; // Element nodes
          }

          for (int j = 0; j < nnode_elem; j = j + 1) {
            for (int v = 0; v <= 1; v = v + 1) {
              int m = eNodes[j];
              eCoord[j * 2 + v] = Coord[m * 2 + v];
            }
          }

          int *gDof = new int[nnode_elem]; // used to constuct scatter matrix
          int gDofNode;
          for (int j = 0; j < nnode_elem; j = j + 1) {

            // global dof for node j
            int m = eNodes[j];
            int NoDf = nNodeDof[j] + 1;

            for (int k = 1; k < NoDf; k = k + 1) {
              int kk = k - 1;
              gDofNode = globDof[m * 2 + k];
            }

            gDof[j] = gDofNode;
          }

          //- Local stiffnessmatrix, Ke, is found
          //----- Element stiffness matrix, Ke, by Gauss integration -----------

          double *Ke = new double[nnode_elem * nnode_elem];

          zeros(Ke, nnode_elem, nnode_elem);

          double *DetJ =
              new double[gauss * gauss]; // For storing the determinants of J

          double *XX = new double[nnode_elem];
          double *YY = new double[nnode_elem];

          for (int h = 0; h < 4; h = h + 1) {
            XX[h] = eCoord[h * 2 + 0];
            YY[h] = eCoord[h * 2 + 1];
          }

          for (int ii = 0; ii < gauss; ii = ii + 1) {
            for (int jj = 0; jj < gauss; jj = jj + 1) {

              float eta = GP[ii];
              float xi = GP[jj];
              // shape functions matrix
              double *N = new double[4];
              N[0] = 0.25 * (1 - xi) * (1 - eta);
              N[1] = 0.25 * (1 + xi) * (1 - eta);
              N[2] = 0.25 * (1 + xi) * (1 + eta);
              N[3] = 0.25 * (1 - xi) * (1 + eta);
              // derivative (gradient) of the shape functions

              double *GN = new double[4 * 2];
              GN[0 * 4 + 0] = -0.25 * (1 - eta);
              GN[0 * 4 + 1] = 0.25 * (1 - eta);
              GN[0 * 4 + 2] = (1 + eta) * 0.25;
              GN[0 * 4 + 3] = -(1 + eta) * 0.25;

              GN[1 * 4 + 0] = -0.25 * (1 - xi);
              GN[1 * 4 + 1] = -0.25 * (1 + xi);
              GN[1 * 4 + 2] = 0.25 * (1 + xi);
              GN[1 * 4 + 3] = 0.25 * (1 - xi);

              double *J = new double[2 * 2];

              multiply_mat_man(GN, eCoord, J, 2, 2, 4);

              double DetJ;

              DetJ = J[0 * 2 + 0] * J[1 * 2 + 1] - J[0 * 2 + 1] * J[1 * 2 + 0];

              inverse(J, 2);

              double *B = new double[4 * 2];

              multiply_mat_man(J, GN, B, 2, 4, 2);

              double *Bt = new double[2 * 4];

              transpose(B, 2, 4, Bt);

              double *dot = new double[2 * 4];

              multiply_mat_man(Bt, D, dot, 4, 2, 2);

              double *ddot = new double[4 * 4];

              multiply_mat_man(dot, B, ddot, 4, 4, 2);

              for (int o = 0; o < nnode_elem; o = o + 1) {
                for (int x = 0; x < nnode_elem; x = x + 1) {
                  Ke[o * nnode_elem + x] =
                      Ke[o * nnode_elem + x] +
                      ddot[o * 4 + x] * tp * DetJ * W[ii] * W[jj];
                }
              }
            }
          }

          for (int v = 0; v < nnode_elem; v = v + 1) {
            int b = gDof[v];

            for (int d = 0; d < nnode_elem; d = d + 1) {

              int c = gDof[d];

              K[b * nDof2 + c] = K[b * nDof2 + c] + Ke[v * nnode_elem + d];
            }
          }
        }

        //------------------------------------------------------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------END
        //OF THE COMMON
        //PART-----------------------------------------------------------------
        //------------------------------------------------------------------------------------------------------------------------------------------------------------------

        // Compute nodal boundary flux vector --- natural B.C
        // Defined on edges
        int *fluxNodes = new int[nelem_x1];

        for (int v = 0; v < nelem_x1; v = v + 1) {
          fluxNodes[v] = NodeTopo[0 * nelem_x1 + v];
        }

        int nFluxNodes = nelem_x1;

        //----- Defining load ----------------------------
        int qflux = -5000; // Constant flux at right edge of the beam

        double *n_bc = new double[(nFluxNodes - 1) * 4];

        for (int v = 0; v < (nFluxNodes - 1); v = v + 1) {
          n_bc[0 * (nFluxNodes - 1) + v] = fluxNodes[v]; // node 1

          n_bc[2 * (nFluxNodes - 1) + v] = qflux; // flux value at node 1
          n_bc[3 * (nFluxNodes - 1) + v] = qflux; // flux value at node 2
        }

        for (int v = 1; v < nFluxNodes; v = v + 1) {
          int vv = v - 1;
          n_bc[1 * (nFluxNodes - 1) + vv] = fluxNodes[v]; // node 2
        }

        int nbe = nFluxNodes - 1; // Number of elements with flux load

        double *Coordt = new double[nnode * 2];

        transpose(Coord, nnode, 2, Coordt);

        double *xcoord = new double[nnode]; // Nodal coordinates
        double *ycoord = new double[nnode];

        for (int v = 0; v < nnode; v = v + 1) {
          xcoord[v] = Coordt[0 * nnode + v];
          ycoord[v] = Coordt[1 * nnode + v];
        }

        double *f = new double[nDof];

        for (int v = 0; v < nDof; v = v + 1) {
          f[v] = 0;
        }

        double *n_bce = new double[2];

        for (int i = 0; i < nbe; i = i + 1) {

          double *fq = new double[2];
          zeros_vect(fq, 2); // initialize the nodal source vector

          int node1 = n_bc[0 * (nFluxNodes - 1) + i]; // first node
          int node2 = n_bc[1 * (nFluxNodes - 1) + i]; // second node

          for (int v = 2; v < 4; v = v + 1) {
            int vv = v - 2;
            n_bce[vv] = n_bc[v * (nFluxNodes - 1) + i]; // flux value at an edge
          }

          double x1 = xcoord[node1]; // x coord of the first node
          double y1 = ycoord[node1]; // y coord of the first node
          double x2 = xcoord[node2]; // x coord of the first node
          double y2 = ycoord[node2]; // y coord of the second node

          double leng =
              pow(pow((x2 - x1), 2) + pow((y2 - y1), 2), 0.5); // edge length
          double detJ = leng / 2;                              // 1D Jacobian

          // integrate in xi direction (1D integration)

          for (int j = 0; j < gauss; j = j + 1) {

            double xi = GP[j]; // 1D  shape functions in parent domain
            double xii = 0.5 * (1 - xi);
            double xiii = 0.5 * (1 + xi);
            double N[2] = {xii, xiii};

            double flux = 0;

            for (int o = 0; o < 2; o = o + 1) {

              flux = flux + N[o] * n_bce[o];
            }

            fq[0] = fq[0] + W[j] * N[0] * flux * detJ * tp;
            fq[1] = fq[1] + W[j] * N[1] * flux * detJ * tp;
          }
          //  define flux as negative integrals
          fq[0] = -fq[0];
          fq[1] = -fq[1];

          f[node1] = f[node1] + fq[0];
          f[node2] = f[node2] + fq[1];
        }

        for (int i = nbe - 1; i < nbe; i = i + 1) {

          double *fq = new double[2];
          zeros_vect(fq, 2); // initialize the nodal source vector

          int node1 = n_bc[0 * (nFluxNodes - 1) + i]; // first node
          int node2 = n_bc[1 * (nFluxNodes - 1) + i]; // second node

          for (int v = 2; v < 4; v = v + 1) {
            int vv = v - 2;
            n_bce[vv] = n_bc[v * (nFluxNodes - 1) + i]; // flux value at an edge
          }

          double x1 = xcoord[node1]; // x coord of the first node
          double y1 = ycoord[node1]; // y coord of the first node
          double x2 = xcoord[node2]; // x coord of the first node
          double y2 = ycoord[node2]; // y coord of the second node

          double leng =
              pow(pow((x2 - x1), 2) + pow((y2 - y1), 2), 0.5); // edge length
          double detJ = leng / 2;                              // 1D Jacobian

          // integrate in xi direction (1D integration)

          for (int j = 0; j < gauss; j = j + 1) {

            double xi = GP[j]; // 1D  shape functions in parent domain
            double xii = 0.5 * (1 - xi);
            double xiii = 0.5 * (1 + xi);
            double N[2] = {xii, xiii};

            double flux = 0;

            for (int o = 0; o < 2; o = o + 1) {

              flux = flux + N[o] * n_bce[o];
            }

            fq[0] = fq[0] + W[j] * N[0] * flux * detJ * tp;
            fq[1] = fq[1] + W[j] * N[1] * flux * detJ * tp;
          }
          //  define flux as negative integrals
          fq[0] = -fq[0];
          fq[1] = -fq[1];

          // f[node1] = f[node1]+fq[0];
          f[node2] = f[node2] + fq[1];
        }

        //----- Apply boundary conditions ----------------- Essential
        //B.C.-------------------

        int *TempNodes = new int[q]; // Nodes at the left edge of the beam

        for (int v = 0; v < q; v = v + 1) {

          TempNodes[v] = NodeTopo[v * nelem_x1 + 0];
        }

        //------------------------------------------------

        int nTempNodes = q; // Number of nodes with temp BC

        double *BC =
            new double[2 *
                       nTempNodes]; // initialize the nodal temperature vector

        zeros(BC, nTempNodes, 2);

        int T0 = -20; // Temperature at boundary

        for (int v = 0; v < nTempNodes; v = v + 1) {
          BC[v * 2 + 0] = TempNodes[v];
          BC[v * 2 + 1] = T0;
        }

        //---------------- Assembling global "Force" vector ------------

        double *OrgDof = new double[nDof]; //  Original DoF number

        zeros_vect(OrgDof, nDof);

        double *T = new double[nDof]; // initialize nodal temperature vector

        int rDof = nDof; // Reduced number of DOF

        int ind[nTempNodes];

        for (int v = 0; v < nTempNodes; v = v + 1) {
          ind[v] = BC[v * 2 + 0];
        }

        for (int v = 0; v < nTempNodes; v = v + 1) {
          int t = ind[v];
          OrgDof[t] = -1;
        }

        for (int v = 0; v < nTempNodes; v = v + 1) {
          int t = ind[v];
          OrgDof[t] = -1;
        }

        for (int v = 0; v < nTempNodes; v = v + 1) {
          int t = ind[v];
          T[t] = BC[v * 2 + 1];
        }

        rDof = rDof - nTempNodes;

        int RedDof[rDof];
        int counter1 = 0;

        for (int v = 0; v < nDof; v = v + 1) {
          if (OrgDof[v] == 0) {
            OrgDof[v] = counter1;
            RedDof[counter1] = v;
            counter1 = counter1 + 1;
          }
        }

        //-------------------------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------------------------

        // Partition matrices

        int *mask_E = new int[nDof];

        for (int v = 0; v < nDof; v = v + 1) {
          for (int b = 0; b < nTempNodes; b = b + 1) {
            float bb = TempNodes[b];
            if (v == bb) {
              mask_E[v] = 1;
              break;
            } else {
              mask_E[v] = 0;
            }
          }
        }

        //-------------------------------------------------------------------------------------------------------------------------------------------------

        int mask_Ec = 0;

        for (int v = 0; v < nDof; v = v + 1) {
          mask_Ec = mask_Ec + mask_E[v];
        }
        //-----------------------------------------------

        int *mask_EE = new int[mask_Ec];

        int co = 0;

        for (int v = 0; v < nDof; v = v + 1) {

          if (mask_E[v] == 1) {

            mask_EE[co] = v;
            co = co + 1;
          }
        }

        //------------------------------------

        double *T_E = new double[mask_Ec];

        for (int v = 0; v < mask_Ec; v = v + 1) {
          int bb = mask_EE[v];
          T_E[v] = T[bb];
        }

        //-------------------------------------------------

        int fmask_Ec = 0;

        for (int v = 0; v < nDof; v = v + 1) {
          if (mask_E[v] == 0) {
            fmask_Ec = fmask_Ec + 1;
          }
        }

        //------------------------------------

        int *fmask_EE = new int[fmask_Ec];

        int fco = 0;

        for (int v = 0; v < nDof; v = v + 1) {

          if (mask_E[v] == 0) {

            fmask_EE[fco] = v;
            fco = fco + 1;
          }
        }

        //--------------------------------------------------------------------------------------------------------

        double *K_EE = new double[mask_Ec * mask_Ec];

        zeros(K_EE, mask_Ec, mask_Ec);

        for (int v = 0; v < mask_Ec; v = v + 1) {
          int b = mask_EE[v];

          for (int d = 0; d < mask_Ec; d = d + 1) {

            int c = mask_EE[d];
            K_EE[v * mask_Ec + d] = K[b * nDof2 + c];
          }
        }

        //---------------------------------------------------------------------------------------------------------------------------

        double *K_FF = new double[fmask_Ec * fmask_Ec];

        zeros(K_FF, fmask_Ec, fmask_Ec);

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          int b = fmask_EE[v];

          for (int d = 0; d < fmask_Ec; d = d + 1) {

            int c = fmask_EE[d];
            K_FF[v * fmask_Ec + d] = K[b * nDof2 + c];
          }
        }

        //---------------------------------------------------------------------------------------------

        double *K_EF = new double[fmask_Ec * mask_Ec];

        zeros(K_EF, mask_Ec, fmask_Ec);

        for (int v = 0; v < mask_Ec; v = v + 1) {
          int b = mask_EE[v];

          for (int d = 0; d < fmask_Ec; d = d + 1) {

            int c = fmask_EE[d];

            K_EF[v * fmask_Ec + d] = K[b * nDof2 + c];
          }
        }

        // solve for d_F

        double *rhs = new double[fmask_Ec];

        double *K_EFt = new double[mask_Ec * fmask_Ec];

        transpose(K_EF, mask_Ec, fmask_Ec, K_EFt);

        double *prod = new double[fmask_Ec];

        for (int o = 0; o < fmask_Ec; o = o + 1) {

          prod[o] = 0;
          for (int k = 0; k < mask_Ec; k = k + 1) {
            prod[o] = prod[o] + K_EFt[o * mask_Ec + k] * T_E[k];
          }
        }

        double *f_F = new double[fmask_Ec];

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          int bb = fmask_EE[v];
          f_F[v] = f[bb];
        }

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          rhs[v] = f_F[v] - prod[v];
        }

        // The resolution is based on the conjugate gradient method
        // The method is explained with more details at rank=1

        //---------------------------------RESOLVE
        //SYSTEM---------------------------------

        // Exchange of all the different parameters between the 2 processes

        int mask_Eca = mask_Ec;
        MPI_Send(&mask_Eca, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

        int fmask_Eca = fmask_Ec;
        MPI_Send(&fmask_Eca, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

        int nDof1 = nDof;
        MPI_Send(&nDof1, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

        int taiTF;
        MPI_Recv(&taiTF, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // K_FFa
        double *K_FFa = new double[fmask_Eca * fmask_Eca];

        for (int i = 0; i < fmask_Eca; i = i + 1) {
          for (int j = 0; j < fmask_Eca; j = j + 1) {

            K_FFa[i * fmask_Eca + j] = K_FF[i * fmask_Eca + j];
          }
        }

        double *rhsa = new double[fmask_Eca];
        for (int i = 0; i < fmask_Eca; i = i + 1) {
          rhsa[i] = rhs[i];
        }

        MPI_Send(rhsa, fmask_Eca, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);

        // Defintion of a random initial vector x0b
        double *x0a = new double[fmask_Eca];

        for (int i = 0; i < fmask_Eca; i = i + 1) {
          x0a[i] = 2;
        }

        double *y0a = new double[fmask_Eca];

        multiply_vect_man(K_FFa, x0a, y0a, fmask_Eca, fmask_Eca);

        double *r0 = new double[taiTF];

        MPI_Recv(r0, taiTF, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        for (int i = 0; i < fmask_Eca; i = i + 1) {

          r0[i] = r0[i] - y0a[i];
        }

        double *pa = new double[fmask_Eca];

        for (int i = 0; i < fmask_Eca; i = i + 1) {
          pa[i] = r0[i];
        }

        MPI_Send(r0, taiTF, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);

        //-------------------------------------------------------------------

        int k = 0;

        double *xka = new double[fmask_Eca];

        for (int i = 0; i < fmask_Eca; i = i + 1) {

          xka[i] = x0a[i];
        }

        int limit;

        MPI_Recv(&limit, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        while (k < limit) {

          double alpha;
          double *ga = new double[fmask_Eca];

          multiply_vect_man(K_FFa, pa, ga, fmask_Eca, fmask_Eca);

          double ca = 0;

          for (int i = 0; i < fmask_Eca; i = i + 1) {

            ca = ca + pa[i] * ga[i];
          }
          MPI_Send(&ca, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);

          MPI_Recv(&alpha, 1, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);

          for (int i = 0; i < fmask_Eca; i = i + 1) {

            xka[i] = xka[i] + alpha * pa[i];
          }

          double *Aa = new double[fmask_Eca];
          double *Atotal = new double[taiTF];
          double *r00 = new double[taiTF];
          double beta;

          multiply_vect_man(K_FFa, pa, Aa, fmask_Eca, fmask_Eca);

          MPI_Send(Aa, fmask_Eca, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);

          MPI_Recv(r00, taiTF, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);

          MPI_Recv(&beta, 1, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);

          for (int i = 0; i < fmask_Eca; i = i + 1) {

            pa[i] = r00[i] + beta * pa[i];
          }

          MPI_Recv(r0, taiTF, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
          MPI_Recv(&k, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        for (int v = 0; v < mask_Ec; v = v + 1) {
          int bb = mask_EE[v];
          T[bb] = T_E[v];
        }

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          int bb = fmask_EE[v];
          T[bb] = xka[v];
        }

        // Definition of the values to construct the array T at the end

        int lenn;
        MPI_Recv(&lenn, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int fmask_Ecb;
        MPI_Recv(&fmask_Ecb, 1, MPI_INT, 1, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        mask_Eca0 = mask_Eca;
        fmask_Eca0 = fmask_Eca;
        nDof0 = nDof1;

        fmask_Ecb0 = fmask_Ecb;
        lenn0 = lenn;

        for (int i = 0; i < nDof1; i = i + 1) {

          Ttransi[i] = T[i];
        }
      }

      if (rank == 1) {

        int nelem_x1;

        if (nelem_x % 2 == 0) {

          nelem_x1 = (nelem_x + 2) / 2;
        }

        else {

          nelem_x1 = (nelem_x + 1) / 2;
        }

        int nelem_x2;

        if (nelem_x % 2 == 0) {

          nelem_x2 = (nelem_x + 2) / 2;
        }

        else {

          nelem_x2 = (nelem_x + 1) / 2;
        }

        int diffnelem_x2;

        diffnelem_x2 = (nelem_x) + 1 - nelem_x2 + 1;

        int *NodeTopo = new int[diffnelem_x2 * q];

        zeros_int(NodeTopo, q, diffnelem_x2);

        for (int colnr = nelem_x2 - 1; colnr < nelem_x + 1; colnr = colnr + 1) {
          for (int i = 0; i <= nelem_y; i = i + 1) {
            int colnrr = colnr - nelem_x2 + 1;

            NodeTopo[i * diffnelem_x2 + colnrr] = (colnr) * (nelem_y + 1) + i;
          }
        }

        //----- Calculation of topology matrix ElemNode
        //------------------------------

        int tai = (nelem_x1 - 1) * nelem_y;

        int taille = nelem - tai;

        int *ElemNode = new int[5 * taille]; // Element connectivity

        zeros_int(ElemNode, (nelem - tai), 5);

        int elemnr = (nelem_x1 - 1) * (nelem_y);

        for (int colnr = 0; colnr < diffnelem_x2 - 1; colnr = colnr + 1) {
          for (int rownr = 0; rownr < nelem_y; rownr = rownr + 1) {

            int elemnrr = elemnr - (nelem_x1 - 1) * (nelem_y);

            ElemNode[elemnrr * 5 + 0] = elemnr;
            ElemNode[elemnrr * 5 + 4] =
                NodeTopo[(rownr + 1) * diffnelem_x2 + colnr]; // Lower left node
            ElemNode[elemnrr * 5 + 3] =
                NodeTopo[(rownr + 1) * diffnelem_x2 +
                         (colnr + 1)]; // Lower right node
            ElemNode[elemnrr * 5 + 2] =
                NodeTopo[rownr * diffnelem_x2 +
                         (colnr + 1)]; // Upper right node
            ElemNode[elemnrr * 5 + 1] =
                NodeTopo[rownr * diffnelem_x2 + colnr]; // upper left node
            elemnr = elemnr + 1;
          }
        }

        tail_Elem1 = (nelem - tai);
        MPI_Send(&tail_Elem1, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Recv(&tail_Elem0, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        for (int i = 0; i < taille; i = i + 1) {
          for (int j = 0; j < 5; j = j + 1) {
            ElemNodeTransi[5 * i + j] = ElemNode[5 * i + j];
          }
        }

        double *ElemX = new double[nnode_elem * (nelem - tai)];
        double *ElemY = new double[nnode_elem * (nelem - tai)];

        zeros(ElemX, (nelem - tai), nnode_elem);
        zeros(ElemY, (nelem - tai), nnode_elem);

        double *eNodes = new double[4];

        zeros_vect(eNodes, 4);

        double *eCoord = new double[2 * 4];

        zeros(eCoord, 4, 2);

        for (int i = 0; i < (nelem - tai); i = i + 1) {
          for (int g = 1; g < 5; g = g + 1) {
            int gg = g - 1;
            eNodes[gg] = ElemNode[i * 5 + g]; // Element nodes
          }

          for (int j = 0; j < nnode_elem; j = j + 1) {
            for (int v = 0; v <= 1; v = v + 1) {

              int m = eNodes[j];
              eCoord[j * 2 + v] = Coord[m * 2 + v];
            }
          }

          for (int h = 0; h < nnode_elem; h = h + 1) {
            ElemX[i * nnode_elem + h] = eCoord[h * 2 + 0];
            ElemY[i * nnode_elem + h] = eCoord[h * 2 + 1];
          }
        }

        //----- --------------------Generate global dof numbers
        //--------------------------------

        int nnode1 = (nelem_x1) * (nelem_y + 1);

        int nnode2 = nnode - nnode1;

        int nnode_min = ElemNode[0 * 5 + 1];

        for (int v = 2; v < 4; v = v + 1) {
          if (nnode_min > ElemNode[0 * 5 + v + 1]) {
            nnode_min = ElemNode[0 * 5 + v + 1];
          }
        }

        int nnode_min_tai = nnode - nnode_min;

        int *globDof = new int[2 * (nnode_min_tai)]; // nDof/node, Dof number

        zeros_int(globDof, nnode_min_tai, 2);

        // int* Top=new int[4*taille];

        int nNode;
        // int* globNodes=new int[4*taille];

        // int val_max=MPI_Recev(...)+1; it is supposed to be 36;

        for (int j = 0; j < taille; j = j + 1) {

          // for(int s=1;s<5;s=s+1){
          //
          // int ss=s-1;
          // globNodes[j*4+ss]=ElemNode[j*5+s];  //Global node numbers of
          // element nodes
          //}

          for (int k = 0; k < nnode_elem; k = k + 1) {
            nNode = ElemNode[j * 5 + (k + 1)];
            nNode = nNode - nnode_min;

            // if the already existing ndof of the present node is less than
            // the present elements ndof then replace the ndof for that node

            if (nNode >= 0) {
              if (globDof[nNode * 2 + 0] < nNodeDof[k]) {
                globDof[nNode * 2 + 0] = nNodeDof[k];
              }
            }
          }
        }

        // counting the global dofs and inserting in globDof

        int nDof = nnode_min;

        nDof2 = nDof;

        tail_T = nnode_min_tai;

        int eDof;
        for (int j = 0; j < nnode_min_tai; j = j + 1) {

          eDof = globDof[j * 2 + 0];

          for (int k = 0; k < eDof; k = k + 1) {
            globDof[j * 2 + (k + 1)] = nDof2;
            nDof2 = nDof2 + 1;
          }
        }

        MPI_Send(&nDof2, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);

        //---------------------------------------------Assembly of global
        //stiffness matrix K ------------------------------
        //---------------------------------------------------- Gauss-points and
        //weights -----------------------------------

        int gauss = gaussorder; // Gauss order

        // Points
        double *GP = new double[2];

        GP[0] = -1 / pow(3, 0.5);
        GP[1] = 1 / pow(3, 0.5);

        // Weights

        int *W = new int[2];

        W[0] = 1;
        W[1] = 1;

        //----- Conductivity matrix D  -----------

        double *D = new double[2 * 2];

        D[0 * 2 + 0] = kxx;
        D[0 * 2 + 1] = kxy;
        D[1 * 2 + 0] = kxy;
        D[1 * 2 + 1] = kyy;

        //----------------------------------------

        double *K =
            new double[nDof2 *
                       nDof2]; // Initiation of global stiffness matrix K

        zeros(K, nDof2, nDof2);

        for (int i = 0; i < taille; i = i + 1) {

          // - data for element i

          for (int g = 1; g < 5; g = g + 1) {
            int gg = g - 1;
            eNodes[gg] = ElemNode[i * 5 + g]; // Element nodes
          }

          for (int j = 0; j < nnode_elem; j = j + 1) {
            for (int v = 0; v <= 1; v = v + 1) {
              int m = eNodes[j];
              eCoord[j * 2 + v] = Coord[m * 2 + v];
            }
          }

          int *gDof = new int[nnode_elem]; // used to constuct scatter matrix
          int gDofNode;

          for (int j = 0; j < nnode_elem; j = j + 1) {

            // global dof for node j

            int m = eNodes[j] - nnode_min;

            int NoDf = nNodeDof[j] + 1;

            for (int k = 1; k < NoDf; k = k + 1) {
              int kk = k - 1;

              gDofNode = globDof[m * 2 + k];
            }

            gDof[j] = gDofNode;
          }

          //- Local stiffnessmatrix, Ke, is found
          //----- Element stiffness matrix, Ke, by Gauss integration -----------

          double *Ke = new double[nnode_elem * nnode_elem];

          zeros(Ke, nnode_elem, nnode_elem);

          double *DetJ =
              new double[gauss * gauss]; // For storing the determinants of J

          double *XX = new double[nnode_elem];
          double *YY = new double[nnode_elem];

          for (int h = 0; h < 4; h = h + 1) {
            XX[h] = eCoord[h * 2 + 0];
            YY[h] = eCoord[h * 2 + 1];
          }

          for (int ii = 0; ii < gauss; ii = ii + 1) {
            for (int jj = 0; jj < gauss; jj = jj + 1) {

              double eta = GP[ii];
              double xi = GP[jj];

              // shape functions matrix

              double *N = new double[4];

              N[0] = 0.25 * (1 - xi) * (1 - eta);
              N[1] = 0.25 * (1 + xi) * (1 - eta);
              N[2] = 0.25 * (1 + xi) * (1 + eta);
              N[3] = 0.25 * (1 - xi) * (1 + eta);

              // derivative (gradient) of the shape functions

              double *GN = new double[4 * 2];

              GN[0 * 4 + 0] = -0.25 * (1 - eta);
              GN[0 * 4 + 1] = 0.25 * (1 - eta);
              GN[0 * 4 + 2] = (1 + eta) * 0.25;
              GN[0 * 4 + 3] = -(1 + eta) * 0.25;

              GN[1 * 4 + 0] = -0.25 * (1 - xi);
              GN[1 * 4 + 1] = -0.25 * (1 + xi);
              GN[1 * 4 + 2] = 0.25 * (1 + xi);
              GN[1 * 4 + 3] = 0.25 * (1 - xi);

              double *J = new double[2 * 2];

              multiply_mat_man(GN, eCoord, J, 2, 2, 4);

              double DetJ;

              DetJ = J[0 * 2 + 0] * J[1 * 2 + 1] - J[0 * 2 + 1] * J[1 * 2 + 0];

              inverse(J, 2);

              double *B = new double[4 * 2];

              multiply_mat_man(J, GN, B, 2, 4, 2);

              double *Bt = new double[2 * 4];

              transpose(B, 2, 4, Bt);

              double *dot = new double[2 * 4];

              multiply_mat_man(Bt, D, dot, 4, 2, 2);

              double *ddot = new double[4 * 4];

              multiply_mat_man(dot, B, ddot, 4, 4, 2);

              for (int o = 0; o < nnode_elem; o = o + 1) {
                for (int x = 0; x < nnode_elem; x = x + 1) {
                  Ke[o * nnode_elem + x] =
                      Ke[o * nnode_elem + x] +
                      ddot[o * 4 + x] * tp * DetJ * W[ii] * W[jj];
                }
              }
            }
          }

          for (int v = 0; v < nnode_elem; v = v + 1) {
            int b = gDof[v];

            for (int d = 0; d < nnode_elem; d = d + 1) {

              int c = gDof[d];

              K[b * nDof2 + c] = K[b * nDof2 + c] + Ke[v * nnode_elem + d];
            }
          }
        }

        //----------------------------------------------------------------------------FLUX------------------------------------------------

        //---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        //--------------------------------------------------------------END OF
        //THE COMMON
        //PART---------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        //----------------------------------------------------------------------------FLUX------------------------------------------------

        // Compute nodal boundary flux vector --- natural B.C
        // Defined on edges

        int *fluxNodes = new int[diffnelem_x2];

        for (int v = 0; v < diffnelem_x2; v = v + 1) {
          fluxNodes[v] = NodeTopo[0 * diffnelem_x2 + v];
        }

        int nFluxNodes = diffnelem_x2;

        int prem = fluxNodes[0];

        //----- Defining load ----------------------------
        int qflux = -5000; // Constant flux at right edge of the beam

        double *n_bc = new double[(nFluxNodes - 1) * 4];

        for (int v = 0; v < nFluxNodes - 1; v = v + 1) {
          n_bc[0 * (nFluxNodes - 1) + v] = fluxNodes[v]; // node 1

          n_bc[2 * (nFluxNodes - 1) + v] = qflux; // flux value at node 1
          n_bc[3 * (nFluxNodes - 1) + v] = qflux; // flux value at node 2
        }

        for (int v = 1; v < nFluxNodes; v = v + 1) {
          int vv = v - 1;
          n_bc[1 * (nFluxNodes - 1) + vv] = fluxNodes[v]; // node 2
        }

        int nbe = nFluxNodes - 1; // Number of elements with flux load

        double *Coordt = new double[nnode * 2];

        transpose(Coord, nnode, 2, Coordt);

        double *xcoord = new double[nnode]; // Nodal coordinates
        double *ycoord = new double[nnode];

        for (int v = 0; v < nnode; v = v + 1) {
          xcoord[v] = Coordt[0 * nnode + v];
          ycoord[v] = Coordt[1 * nnode + v];
        }

        double *f = new double[nnode_min_tai];

        for (int v = 0; v < nnode_min_tai; v = v + 1) {
          f[v] = 0;
        }

        double *n_bce = new double[2];

        for (int i = 0; i < nbe; i = i + 1) {

          double *fq = new double[2];
          zeros_vect(fq, 2); // initialize the nodal source vector

          int node1 = n_bc[0 * (nFluxNodes - 1) + i]; // first node
          int node2 = n_bc[1 * (nFluxNodes - 1) + i]; // second node

          for (int v = 2; v < 4; v = v + 1) {
            int vv = v - 2;
            n_bce[vv] = n_bc[v * (nFluxNodes - 1) + i]; // flux value at an edge
          }

          double x1 = xcoord[node1]; // x coord of the first node
          double y1 = ycoord[node1]; // y coord of the first node
          double x2 = xcoord[node2]; // x coord of the first node
          double y2 = ycoord[node2]; // y coord of the second node

          double leng =
              pow(pow((x2 - x1), 2) + pow((y2 - y1), 2), 0.5); // edge length
          double detJ = leng / 2;                              // 1D Jacobian

          // integrate in xi direction (1D integration)

          for (int j = 0; j < gauss; j = j + 1) {

            double xi = GP[j]; // 1D  shape functions in parent domain
            double xii = 0.5 * (1 - xi);
            double xiii = 0.5 * (1 + xi);
            double N[2] = {xii, xiii};

            double flux = 0;

            for (int o = 0; o < 2; o = o + 1) {

              flux = flux + N[o] * n_bce[o];
            }

            fq[0] = fq[0] + W[j] * N[0] * flux * detJ * tp;
            fq[1] = fq[1] + W[j] * N[1] * flux * detJ * tp;
          }
          //  define flux as negative integrals
          fq[0] = -fq[0];
          fq[1] = -fq[1];

          int diff = node2 - node1 - 1;

          int dnode1 = node1 - prem;
          int dnode2 = node2 - prem;

          f[dnode1] = f[dnode1] + fq[0];
          f[dnode2] = f[dnode2] + fq[1];
        }

        for (int i = 0; i < 1; i = i + 1) {

          double *fq = new double[2];
          zeros_vect(fq, 2); // initialize the nodal source vector

          int node1 = n_bc[0 * (nFluxNodes - 1) + i]; // first node
          int node2 = n_bc[1 * (nFluxNodes - 1) + i]; // second node

          for (int v = 2; v < 4; v = v + 1) {
            int vv = v - 2;
            n_bce[vv] = n_bc[v * (nFluxNodes - 1) + i]; // flux value at an edge
          }

          double x1 = xcoord[node1]; // x coord of the first node
          double y1 = ycoord[node1]; // y coord of the first node
          double x2 = xcoord[node2]; // x coord of the first node
          double y2 = ycoord[node2]; // y coord of the second node

          double leng =
              pow(pow((x2 - x1), 2) + pow((y2 - y1), 2), 0.5); // edge length
          double detJ = leng / 2;                              // 1D Jacobian

          // integrate in xi direction (1D integration)

          for (int j = 0; j < gauss; j = j + 1) {

            double xi = GP[j]; // 1D  shape functions in parent domain
            double xii = 0.5 * (1 - xi);
            double xiii = 0.5 * (1 + xi);
            double N[2] = {xii, xiii};

            double flux = 0;

            for (int o = 0; o < 2; o = o + 1) {

              flux = flux + N[o] * n_bce[o];
            }

            fq[0] = fq[0] + W[j] * N[0] * flux * detJ * tp;
            fq[1] = fq[1] + W[j] * N[1] * flux * detJ * tp;
          }
          //  define flux as negative integrals
          fq[0] = -fq[0];
          fq[1] = -fq[1];

          int diff = node2 - node1 - 1;

          int dnode1 = node1 - prem;
          int dnode2 = node2 - prem;

          f[dnode1] = f[dnode1] + fq[0];
          // f[dnode2] = f[dnode2] + fq[1];
        }

        //----- Apply boundary conditions ----------------- Essential
        //B.C.-------------------

        int *TempNodes =
            new int[diffnelem_x2]; // Nodes at the left edge of the beam

        zeros_vect_int(TempNodes, diffnelem_x2);

        //------------------------------------------------

        int nTempNodes = diffnelem_x2; // Number of nodes with temp BC

        double *BC =
            new double[2 *
                       nTempNodes]; // initialize the nodal temperature vector

        zeros(BC, nTempNodes, 2);

        int T0 = -20; // Temperature at boundary

        for (int v = 0; v < nTempNodes; v = v + 1) {
          BC[v * 2 + 0] = TempNodes[v];
          BC[v * 2 + 1] = T0;
        }

        //---------------- Assembling global "Force" vector ------------

        double *OrgDof = new double[nnode_min_tai]; //  Original DoF number

        zeros_vect(OrgDof, nnode_min_tai);

        double *T =
            new double[nnode_min_tai]; // initialize nodal temperature vector

        int rDof = nnode_min_tai; // Reduced number of DOF

        int ind[nTempNodes];

        int min_ind = 0;

        for (int v = 0; v < nTempNodes; v = v + 1) {
          ind[v] = BC[v * 2 + 0];
        }

        for (int v = 0; v < nTempNodes; v = v + 1) {
          int t = ind[v];
          int tt = t - min_ind;
          OrgDof[tt] = -1;
        }

        for (int v = 0; v < nTempNodes; v = v + 1) {
          int t = ind[v];
          int tt = t - min_ind;
          OrgDof[tt] = -1;
        }

        for (int v = 0; v < nTempNodes; v = v + 1) {
          int t = ind[v];
          int tt = t - min_ind;
          T[tt] = BC[v * 2 + 1];
        }

        rDof = rDof - nTempNodes;

        int RedDof[rDof];
        int counter1 = 0;

        for (int v = 0; v < nnode_min_tai; v = v + 1) {
          if (OrgDof[v] == 0) {
            OrgDof[v] = counter1;
            RedDof[counter1] = v;
            counter1 = counter1 + 1;
          }
        }

        T[0] = 0; // To balance the fact that T initially contains the value -20
                  // as its first value

        // Partition matrices

        int *mask_E = new int[nnode_min_tai];

        for (int v = 0; v < nnode_min_tai; v = v + 1) {
          for (int b = 0; b < diffnelem_x2; b = b + 1) {
            float bb = TempNodes[b];
            bb = bb - min_ind;

            if (v == bb) {
              mask_E[v] = 1;
              break;
            } else {
              mask_E[v] = 0;
            }
          }
        }

        mask_E[0] = 0; // To balance the fact that mask_E initially contains the
                       // value -20 as its first value

        int mask_Ec = 0;

        for (int v = 0; v < nnode_min_tai; v = v + 1) {
          mask_Ec = mask_Ec + mask_E[v];
        }

        int *mask_EE = new int[mask_Ec];

        int co = 0;

        for (int v = 0; v < nnode_min_tai; v = v + 1) {

          if (mask_E[v] == 1) {

            mask_EE[co] = v;
            co = co + 1;
          }
        }

        // T_E is Pointless here for the next calculations

        // double* T_E=new double[mask_Ec];
        //
        // for(int v=0;v<mask_Ec;v=v+1){
        //    int bb=mask_EE[v];
        //    T_E[v]=T[bb];
        //}

        //--------------------------------------------------------------------------------------------------------------------------------------------

        int fmask_Ec = 0;

        for (int v = 0; v < nnode_min_tai; v = v + 1) {
          if (mask_E[v] == 0) {
            fmask_Ec = fmask_Ec + 1;
          }
        }

        int *fmask_EE = new int[fmask_Ec];

        int fco = 0;

        for (int v = 0; v < nnode_min_tai; v = v + 1) {

          if (mask_E[v] == 0) {

            fmask_EE[fco] = v;
            fco = fco + 1;
          }
        }

        double *f_F = new double[fmask_Ec];

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          int bb = fmask_EE[v];
          f_F[v] = f[bb];
        }

        //---------------------------------------------------------------------------------------------------------------------------

        min_ind = nDof2 - fmask_Ec;

        double *K_FF = new double[fmask_Ec * fmask_Ec];

        zeros(K_FF, fmask_Ec, fmask_Ec);

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          int b = fmask_EE[v] + min_ind;

          for (int d = 0; d < fmask_Ec; d = d + 1) {

            int c = fmask_EE[d] + min_ind;
            K_FF[v * fmask_Ec + d] = K[b * nDof2 + c];
          }
        }

        // solve for d_F

        double *rhs = new double[fmask_Ec];

        for (int v = 0; v < fmask_Ec; v = v + 1) {
          rhs[v] = f_F[v];
        }

        //------------------------------RESOLVE
        //SYSTEM---------------------------------------

        // We exchange between the 2 processes, all the dimesions of the arrays
        // and matrices

        // Recpetion of mask_Eca
        int mask_Eca;
        MPI_Recv(&mask_Eca, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        // Reception of mask_Eca

        int fmask_Eca;
        MPI_Recv(&fmask_Eca, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        // Reception of nDof1
        int nDof1;
        MPI_Recv(&nDof1, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Sending of taiTF
        int taiTF = nDof2 - mask_Eca;
        MPI_Send(&taiTF, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);

        // Reception of rhsa

        double *rhsa = new double[fmask_Eca];

        MPI_Recv(rhsa, fmask_Eca, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        // fmask_Ecb

        int fmask_Ecb = fmask_Ec;

        // rhsb

        double *rhsb = new double[fmask_Ecb];
        for (int i = 0; i < fmask_Ecb; i = i + 1) {
          rhsb[i] = rhs[i];
        }

        // K_FFb

        double *K_FFb = new double[fmask_Ecb * fmask_Ecb];

        for (int i = 0; i < fmask_Ecb; i = i + 1) {
          for (int j = 0; j < fmask_Ecb; j = j + 1) {

            K_FFb[i * fmask_Ecb + j] = K_FF[i * fmask_Ecb + j];
          }
        }

        // Initial matrices and values

        double *r0 = new double[taiTF];

        double *pb = new double[fmask_Ecb];

        double *b = new double[taiTF];

        double *x0b = new double[fmask_Ecb];

        // Modification of rhsa
        // Here we sum the common points of the rhs and we juxtapose the arrays
        // to get b

        int lenn = fmask_Ecb - (taiTF - fmask_Eca);

        for (int i = fmask_Eca - lenn; i < fmask_Eca; i = i + 1) {

          int ii = i - fmask_Eca + lenn;
          rhsa[i] = rhsa[i] + rhsb[ii];
        }

        // We fill b with rhs total

        for (int i = 0; i < fmask_Eca; i = i + 1) {
          b[i] = rhsa[i];
        }

        // We fill the rest of b

        for (int i = 0; i < fmask_Ecb - lenn; i = i + 1) {

          int ii = i + fmask_Eca;

          b[ii] = rhsb[i + lenn];
        }

        // Defintion of a random initial vector x0b
        // Nonetheless, we don't juxtapose the whole K_FF matrix
        // we use the same x0 which will multiply K_FFa and K_FFb

        for (int i = 0; i < fmask_Ecb; i = i + 1) {
          x0b[i] = 2;
        }

        double *y0b = new double[fmask_Ecb];

        multiply_vect_man(K_FFb, x0b, y0b, fmask_Ecb, fmask_Ecb);

        for (int i = taiTF - fmask_Ecb; i < taiTF; i = i + 1) {
          int ii = i - taiTF + fmask_Ecb;
          r0[i] = b[i] - y0b[ii];
        }

        MPI_Send(r0, taiTF, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

        MPI_Recv(r0, taiTF, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        // We fill p0

        for (int i = taiTF - fmask_Ecb; i < taiTF; i = i + 1) {

          int ii = i - (taiTF - fmask_Ecb);
          pb[ii] = r0[i];
        }

        // To sum-up: only the vectors r0 (or r00) is not split (for linearity
        // problems)
        // Moreover, alpha and beta are computed at rank=1 and sent to rank=0

        int k = 0;

        double *xkb = new double[fmask_Ecb];

        for (int i = 0; i < fmask_Ecb; i = i + 1) {

          xkb[i] = x0b[i];
        }

        int limit = 500;
        MPI_Send(&limit, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);

        while (k < limit) {

          double alpha = 0;

          for (int i = 0; i < taiTF; i = i + 1) {

            alpha = alpha + r0[i] * r0[i];
          }

          double *gb = new double[fmask_Ecb];

          multiply_vect_man(K_FFb, pb, gb, fmask_Ecb, fmask_Ecb);

          double cb = 0;

          for (int i = 0; i < fmask_Ecb; i = i + 1) {

            cb = cb + pb[i] * gb[i];
          }

          double ca;

          MPI_Recv(&ca, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          cb = cb + ca;

          alpha = double(alpha) / double(cb);

          MPI_Send(&alpha, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

          for (int i = 0; i < fmask_Ecb; i = i + 1) {

            xkb[i] = xkb[i] + alpha * pb[i];
          }

          double *Ab = new double[fmask_Ecb];
          double *Atotal = new double[taiTF];
          double *Aa = new double[fmask_Eca];

          multiply_vect_man(K_FFb, pb, Ab, fmask_Ecb, fmask_Ecb);

          MPI_Recv(Aa, fmask_Eca, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);

          // We fill Atotal with the entiere multiplication

          for (int i = fmask_Eca - lenn; i < fmask_Eca; i = i + 1) {

            int ii = i - fmask_Eca + lenn;
            Aa[i] = Aa[i] + Ab[ii];
          }

          for (int i = 0; i < fmask_Eca; i = i + 1) {
            Atotal[i] = Aa[i];
          }

          // We fill the rest of Atotal

          for (int i = 0; i < fmask_Ecb - lenn; i = i + 1) {

            int ii = i + fmask_Eca;

            Atotal[ii] = Ab[i + lenn];
          }

          double *r00 = new double[taiTF];

          for (int i = 0; i < taiTF; i = i + 1) {

            r00[i] = r0[i] - alpha * Atotal[i];
          }

          MPI_Send(r00, taiTF, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

          double beta = 0;

          for (int i = 0; i < taiTF; i = i + 1) {

            beta = beta + r00[i] * r00[i];
          }

          double gg = 0;

          for (int i = 0; i < taiTF; i = i + 1) {

            gg = gg + r0[i] * r0[i];
          }

          beta = double(beta) / double(gg);

          MPI_Send(&beta, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

          for (int i = 0; i < fmask_Ecb; i = i + 1) {
            int ii = taiTF - fmask_Ecb + i;
            pb[i] = r00[ii] + beta * pb[i];
          }

          for (int i = 0; i < taiTF; i = i + 1) {

            r0[i] = r00[i];
          }
          MPI_Send(r0, taiTF, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

          k = k + 1;

          MPI_Send(&k, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        }

        // Exchange of the parameters with process 0 to construct the array T at
        // the end

        MPI_Send(&lenn, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(&fmask_Ecb, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);

        fmask_Ecb0 = fmask_Ecb;
        lenn0 = lenn;

        mask_Eca0 = mask_Eca;
        fmask_Eca0 = fmask_Eca;
        nDof0 = nDof1;

        for (int i = 0; i < fmask_Ecb; i = i + 1) {

          Ttransi[i] = xkb[i];
        }
      }

      //---------------------------------------------------------------------------------------------------------------------------------------------
      //---------------------------------------------------------------------------------------------------------------------------------------------
      //---------------------------------------------------------------------------------------------------------------------------------------------

      // Gather the arrays which composed T

      int tail_T2 = 2 * fmask_Ecb0;

      double *Ttotal = new double[tail_T2];

      zeros_vect(Ttotal, tail_T2);

      MPI_Gather(Ttransi, fmask_Ecb0, MPI_DOUBLE, Ttotal, fmask_Ecb0,
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);

      // Gather the ElemNodes matrices

      int tail_Elem2 = tail_Elem1 * 2;

      int *ElemNodeTotal = new int[5 * tail_Elem2];

      MPI_Gather(ElemNodeTransi, 5 * tail_Elem1, MPI_INT, ElemNodeTotal,
                 5 * tail_Elem1, MPI_INT, 0, MPI_COMM_WORLD);

      if (rank == 0) {

        double *T1 = new double[nDof0];
        double *T2 = new double[fmask_Ecb0];

        for (int i = 0; i < nDof0; i = i + 1) {
          T1[i] = Ttotal[i];
        }

        for (int i = fmask_Ecb0; i < tail_T2; i = i + 1) {

          int ii = i - fmask_Ecb0;

          T2[ii] = Ttotal[i];
        }

        double *T = new double[nDof2];

        // We sum the common points in T1
        for (int i = nDof0 - lenn0; i < nDof0; i = i + 1) {

          int ii = i - nDof0 + lenn0;
          T1[i] = T1[i] + T2[ii];

          // To tackle the problem of Ta and Tb taking common values at the
          // border
          // we divide by 2 (I assume the slight difference with the Python
          // simulation might come from here)

          T1[i] = T1[i] / 2.0;
        }

        // We add T1 to T2
        for (int i = 0; i < nDof0; i = i + 1) {
          T[i] = T1[i];
        }

        // We add the truncated T2 to T
        for (int i = 0; i < fmask_Ecb0 - lenn0; i = i + 1) {

          int ii = i + nDof0;

          T[ii] = T2[i + lenn0];
        }

        // ElemNodes

        int *ElemNode = new int[5 * nelem];

        for (int i = 0; i < tail_Elem0; i = i + 1) {
          for (int j = 0; j < 5; j = j + 1) {

            ElemNode[i * 5 + j] = ElemNodeTotal[i * 5 + j];
          }
        }

        int diff_tail_Elem = tail_Elem1 - tail_Elem0;

        for (int i = tail_Elem1; i < tail_Elem2; i = i + 1) {
          for (int j = 0; j < 5; j = j + 1) {

            int ii = i - diff_tail_Elem;

            ElemNode[ii * 5 + j] = ElemNodeTotal[i * 5 + j];
          }
        }

        //-----------------------------------------------Create
        //points-----------------------------------------------------

        double *points = new double[3 * nnode];

        for (int v = 0; v < nnode; v = v + 1) {
          for (int b = 0; b < 3; b = b + 1) {

            if (b != 2) {
              points[v * 3 + b] = Coord[v * 2 + b];

            }

            else {
              points[v * 3 + b] = 0;
            }
          }
        }

        //--------------------------------------------Create
        //cells---------------------------------------------------------

        int *cells = new int[4 * nelem];

        for (int i = 0; i < nelem; i = i + 1) {

          for (int j = 0; j < 4; j = j + 1) {

            int jj = j + 1;

            cells[i * 4 + j] = ElemNode[i * 5 + jj];
          }
        }

        //---------------Create VTK
        //file----------------------------------------------------------------

        // Create intro

        ofstream vOut("datap3.vtk", ios::out | ios::trunc);
        vOut << "# vtk DataFile Version 4.0" << endl;
        vOut << "vtk output" << endl;
        vOut << "ASCII" << endl;
        vOut << "DATASET UNSTRUCTURED_GRID" << endl;

        // Print points

        vOut << "POINTS"
             << " " << nnode << " "
             << "double" << endl;
        for (int v = 0; v < nnode; v = v + 1) {
          for (int b = 0; b < 3; b = b + 1) {
            vOut << points[v * 3 + b] << " ";
          }
        }
        vOut << endl;

        // print cells

        int total_num_cells = nelem;
        int total_num_idx = 5 * nelem;

        vOut << "CELLS"
             << " " << total_num_cells << " " << total_num_idx << endl;

        // Creation of keys_cells

        int *keys_cells = new int[5 * nelem];

        for (int i = 0; i < nelem; i = i + 1) {
          keys_cells[i * 5 + 0] = 4;
          for (int j = 1; j < 5; j = j + 1) {
            int jj = j - 1;
            keys_cells[i * 5 + j] = cells[i * 4 + jj];
          }
        }

        // print keys_cells

        for (int i = 0; i < total_num_cells; i = i + 1) {

          for (int j = 0; j < 5; j = j + 1) {

            vOut << keys_cells[i * 5 + j] << " ";
          }
          vOut << endl;
        }

        vOut << "CELL_TYPES"
             << " " << total_num_cells << endl;

        for (int i = 0; i < total_num_cells; i = i + 1) {

          vOut << 9 << endl;
        }

        // Here we don't create "point_data" as we directly use T and
        // len(T)=nDof

        // Print point_data

        int len_points = nnode;

        vOut << "POINT_DATA"
             << " " << len_points << endl;
        vOut << "FIELD FieldData"
             << " "
             << "1" << endl;
        vOut << "disp"
             << " "
             << "1"
             << " " << nDof2 << " "
             << "double" << endl;

        for (int i = 0; i < nDof2; i = i + 1) {
          vOut << T[i] << " ";
        }

        vOut.close();
      }

      MPI_Finalize();

      return 0;
    }
  }
}
