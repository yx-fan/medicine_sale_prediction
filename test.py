class MinStack:
	def __init__(self):
		self.stack = []
		self.min_stack = []
	
	def push(self, x: int) -> None:
		self.stack.append(x)
		if not self.min_stack or x <= self.min_stack[-1]:
			self.min_stack.append(x)
	
	def pop(self) -> None:
		if self.stack:
			top = self.stack.pop()
			if top == self.min_stack[-1]:
				self.min_stack.pop()
	
	def top(self) -> int:
		if self.stack:
			return self.stack[-1]
		else:
			return None
	
	def getMin(self) -> int:
		if self.min_stack:
			return self.min_stack[-1]
		else:
			return None
	
MinStack = MinStack()
MinStack.push(-2)
MinStack.push(0)
MinStack.push(-3)
print(MinStack.getMin()) # return -3
MinStack.pop()
print(MinStack.top()) # return 0
print(MinStack.getMin()) # return -2