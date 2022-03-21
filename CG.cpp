p_0 = r_0 = b - A x_0
for k = 0, 1, ..., maxiter-1
    alpha_k = dot(r_k, r_k) / dot(A p_k, p_k)
    x_(k+1) = x_k + alpha_k p_k
    r_(k+1) = r_k - alpha_k A p_k

    if dot(r_(k+1), r_(k+1)) < tol*tol
       stop

    beta_k  = dot(r_(k+1), r_(k+1)) / dot(r_k, r_k)
    p_(k+1) = r_(k+1) + beta_k p_k


    Here, A is a symmetric positive definite matrix, 
    x_k, r_k and p_k are vectors, 
    dot is the standard l2-inner product, 
    tol is an absolute tolerance for the residual and 
    maxiter is the maximum allowed number of iterations.

Vector<T> p;
Vector<T> r_cur; Vector<T> r_next;
T alpha_k, beta_k;

p = r_cur = A * x;

for (unsigned int k = 0; k < maxIter; ++k) {
    alpha_k = (r_cur * r_cur) / ((A * p) * p);

    x = x + alpha_k * p;

    r_next = r_cur - alpha_k * A * p;

    if (r_next * r_next < tol * tol) break;

    beta_k = (r_next * r_next) / (r_cur * r_cur);
    
    p = r_next + beta_k * p;
    r_cur = r_next;
}