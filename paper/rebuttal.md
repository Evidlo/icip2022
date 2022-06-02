# Rebuttal

Reviewer 0085 mentions that additional tests of failure modes due to model mismatch in the constant-motion assumption would be useful.  We appreciate this suggestion and and agree that an extension of the model to non-constant motion would be valuable. We have been working on this topic by investigating potential sources of model mismatch error that arise in space flight such as rotation, non-rigid motion, jitter, alternative noise models, etc.). We anticipate that our analyses would be included in a journal manuscript on this topic constituting a full treatment of these error sources for full characterization of our algorithm.

Reviewer 1DE1 points out that the same model is used for both data synthesis and in the derivation of the cost function, which may result in an unfair comparison against the other methods.  We plan to address this by adding a realistic amount of jitter and rotation to our test image sequences, which are present in the attitude control systems of all spacecraft.  Estimates of these parameters are available from previous missions using similar hardware. Our initial analysis shows that the addition of this realistic component to our model does not have a significant impact on the effectiveness of our approach; hence it does not alter the significance of the contribution in this manuscript.

We also propose adding an additional registration method to our comparitive analysis: "Astroalign: A Python module for astronomical image registration" [1], intended for rigid-motion registration of noisy star fields.

[1]: Astroalign: A Python module for astronomical image registration. Beroiz, M., Cabral, J. B., & Sanchez, B. Astronomy and Computing, Volume 32, July 2020, 100384.
