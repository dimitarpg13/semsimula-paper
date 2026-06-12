# Training summary — SPLM Multi-Xi OpenWebText

- experiment: OpenWebText SPLM d=256 L=12
- model: ScalarPotentialLMSARFMassLNMultiXi
- corpus: OpenWebText (~200M train tokens)
- params: 16,539,911
- d=256  L=12  v_hidden=1024
- xi_channels=4  alpha_final=[1.016046780932811e-06, 0.42775675654411316, 0.8165659308433533, 0.9860178232192993]
- fixed_gamma: 0.3
- batch_size=8  block_size=512  steps=50000
- seed: 0
- elapsed: 9004s (2.50h)

Final val loss: 5.169325 (ppl 175.80)
Final gamma: 0.3000
Final alpha: [1.016046780932811e-06, 0.42775675654411316, 0.8165659308433533, 0.9860178232192993]
