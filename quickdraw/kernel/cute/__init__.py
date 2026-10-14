"""CuTe DSL kernels for the single-request SM120 path.

These kernels do not import Triton. They are brought up independently before
replacing whole blocks in the engine; see matvec for the W4A16 decode contract.
"""
