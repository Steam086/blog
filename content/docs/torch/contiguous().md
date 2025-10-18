---
title: contiguous
---
x = torch.empty((3, 4))
y = x.movedim(0, 1)
print(x.is_contiguous())
print(y.is_contiguous())


### **transpose()** and **permute()**

- **Do NOT change the underlying memory storage** - they share the same underlying storage with the input tensor [torch.transpose — PyTorch 2.7 documentation](https://docs.pytorch.org/docs/stable/generated/torch.transpose.html)
- They can operate on both contiguous and non-contiguous tensors, but the returned tensor may not be contiguous anymore [Some notes on memory in pytorch. You may have been confused by some… | by rohola zandie | Medium](https://hilbert-cantor.medium.com/some-notes-on-memory-in-pytorch-242cabbff4cc)
- They change the underlying order of elements in the tensor conceptually, but this is achieved through changes to the tensor's stride information, not by moving data in memory [python - Pytorch different outputs between with transpose - Stack Overflow](https://stackoverflow.com/questions/69446402/pytorch-different-outputs-between-with-transpose)

## **movedim()**

- Similar to `transpose()` and `permute()` - it's a **view operation** that doesn't change the actual memory layout
- It only changes the stride and shape metadata to create a new view of the same data


**transpose** is often used in matrix transposition.
**movedim()** is often the most intuitive when you want to "move" dimensions to specific positions, while **permute()** gives you complete control over the final arrangement.

transpose and movedim:
transpose is a swap dim operation, but movedim is a move dim operation.