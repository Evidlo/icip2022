# introduction

  - why is image registration important
    - medical imaging
  - wide range of approaches depending on type of motion, content of scene, changes to object in frame, computational resources available
    - cite zitova & flusser, brown, other survey papers
    
  - this paper focuses on astronomical imaging
    - need to integrate for long periods of time to get sufficient SNR in low light conditions
    - long integration times lead to higher motion blur
    - if separate frames can instead be fused, longer effective integration time can be attained without associated increase in motion blur
    
  - in particular, algorithm was conceived for upcoming VISORS mission, 
    - formation flying cubesat mission studying solar corona at high resolution
    - motion of scene during capture window approximately constant and linear
    
  - designing registration algorithm for particular statistical model can greatly improve registration accuracy
    - cite (optimal correlation at...), (optimal correlation filters for... 1994)
    
  - previous papers which have taken maximum likelihood approach
    - mort & srinath 1988
    - snyder & schulz 1990
    - costa et al 1993
    - roche et al 2000
    - giremus & carfantan 2003
    
  - in next sections
    - provide explicit observation model describing motion of frames and noise
    - description of algorithm and implementation suggestions for faster computation
    - proof of optimality under motion and noise model
    - experimental results of registration error compared to 
    
# algo and observation model

- define variables
- write noise and motion model




# optimality

- why is optimality important -> good performance at low snr (cite?)

# experimental results

# relation to other work

- most closely relates to (sub-pixel image registration with a max...)
  - no closed form - registration must be found with iterative minimization algorithm
  - takes "anchoring" approach, as opposed to "progressive"
  
- exhaustive

# future work

- most subpixel algorithms rely on adhoc interpolation combined with a second search stage for peak
  - apply subpixel sinc interpolation theory to this algorithm for true ML extension to subpixel regime
- integration motion blur properly into model
  - 
  
- image priors? regularization?



# papers/references

- spatial registration of multispectral and multitemporal digital imagery using fast fourier - anuta 1970

- equivalence of ML and weighted least squares - bradley 1973
  - statistics paper on ML vs least squares
- Image registration error variance as a measure of overlay quality 1976
  - applied to image registration
- Maximum Likelihood Image Registration With Subpixel Accuracy - mort 1988


- Sub-pixel image registration with a maximum likelihood estimator - gratadour 2005
  - applies to arbitrary motion model
  - uses numerical minimization on maximum likelihood cost
  - achieves sub-pixel registration using arbitrary interpolation
  
- Maximum-likelihood estimation of an astronomical image from a sequence at low photon levels - guillaume 1998
  - poisson noise - multiframe

- Preliminary Design of a Distributed Telescope CubeSat Formation for Coronal Observations

  
- Correlation filters minimizing peak location errors - kumar 1992
  - proof that cross correlation is optimal solution under additive white gaussian noise
  
- Optimal correlation at low photon levels: study for astronomical images
  - proof of optimality of correlation?
  - assumes noiseless reference is available
  
- Guizar Sicairos
  - course correlation using FFT to find peak
    - refine/upsample correlation with DFT over region of interest
  - does not address optimality
      
- Fundamental Performance Limits in Image Registration
  - defines CRLB for image registration under nyquist setting
  
- brown
- zitova & flusser

- CONSTRAINED, GLOBALLY OPTIMAL, MULTI-FRAME MOTION ESTIMATION
  
- xilinx fft - https://www.xilinx.com/products/intellectual-property/fft.html

- efficient multiframe super-resolution for imagery with lateral shifts
  - computes gradient with respect to shift
  - https://opg.optica.org/ao/fulltext.cfm?uri=ao-53-24-F1&id=297328
  
- projection-based image registration under fixed pattern noise
  - talks about initial calibration step to remove fixed noise
  - talks about windowing to remove circularity assumption**
  - https://ieeexplore.ieee.org/abstract/document/974571
  
- iterative phase correlation for high-precision subpixel registration
  - good references in introduction
  - handles rotation scale w/ log-polar transform
  
       
# software

- https://github.com/bsavitzky/rigidRegistration

# todo

- talk about rotation/scaling using log-polar transform
- address circularity assumption - handled with windowing addressed in other literature
  - "projection-based image registration under fixed pattern noise"
  - iterative phase correlation for high-precision subpixel registration
  - maybe cite Harris 1978?
