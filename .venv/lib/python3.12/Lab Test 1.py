from ctypes import create_unicode_buffer
from operator import truediv
import sys


# Part 1: Linked List

class ListNode(object):
    def __init__(self, val):
        self.val = val
        self.next = None


def insertSortedLL(head: ListNode, val: int): # return the head node
    if head == None:
        return ListNode(val)

    dummy = ListNode(float('-inf'))
    dummy.next = head
    (cur, pre) = (head, dummy)
    while cur is not None:
        if pre.val <= val <= cur.val:
            newNode = ListNode(val)
            newNode.next = cur
            pre.next = newNode
            return dummy.next
        pre = cur
        cur = cur.next
    pre.next = ListNode(val)
    return dummy.next

def alternateMerge(head_1: ListNode, head_2: ListNode):
    if head_1 is None and head_2 is None:
        return None
    dummy = ListNode(-1)
    (cur, cur_1, cur_2) = (dummy, head_1, head_2)
    while cur_1 is not None and cur_2 is not None:
        cur.next = cur_1
        cur_1 = cur_1.next
        cur = cur.next
        cur.next = cur_2
        cur_2 = cur_2.next
        cur = cur.next
    if cur_2 is None:
        cur.next = cur_1
    elif cur_1 is None:
        cur.next = cur_2
    return dummy.next

def moveOddItemsToBack(head: ListNode):
    if head is None:
        return None
    (oddDummy, evenDummy) = (ListNode(-1), ListNode(-1))
    (oddCur, evenCur) = (oddDummy, evenDummy)
    cur = head
    while cur is not None:
        if cur.val % 2 == 1:
            oddCur.next = cur
            oddCur = oddCur.next
        else:
            evenCur.next = cur
            evenCur = evenCur.next
        cur = cur.next
    oddCur.next = None
    evenCur.next = oddDummy.next
    return evenDummy.next

def frontBackSplitLL(head: ListNode):
    if head is None:
        return (None, None)
    (slow, fast) = (head, head)
    while fast.next is not None and fast.next.next is not None:
        slow = slow.next
        fast = fast.next.next
    backHead = slow.next
    slow.next = None
    return (head, backHead)

def moveMaxToFrontLL(head: ListNode):
    dummy = ListNode(-1)
    dummy.next = head
    (cur, maxNode) = (head, head)
    while cur is not None:
        if cur.val > maxNode.val:
            maxNode = cur
        cur = cur.next
    if maxNode == head:
        return head
    (pre, cur) = (dummy, head)
    while cur is not None:
        if cur == maxNode:
            pre.next = cur.next
            cur.next = dummy.next
            dummy.next = cur
            break
        cur = cur.next
        pre = pre.next
    return dummy.next

def removeDuplicates(head: ListNode):
    dummy = ListNode(-1)
    dummy.next = head
    (pre, cur) = (dummy, head)
    while cur.next is not None:
        if cur.val == cur.next.val:
            while cur.next is not None and cur.val == cur.next.val:
                cur = cur.next
            pre.next = cur.next
            cur = cur.next
        else:
            cur = cur.next
            pre = pre.next
    return dummy.next

def duplicate(head: ListNode):
    duplicatedDummy = ListNode(-1)
    (duplicatedPre, cur) = (duplicatedDummy, head)
    while cur is not None:
        if cur.next is not None and cur.next.val == cur.val:
            duplicatedPre.next = cur
            while cur.next is not None and cur.next.val == cur.val:
                cur = cur.next
            duplicatedPre = cur
        cur = cur.next
    duplicatedPre.next = None
    return duplicatedDummy.next


# Part 2: Stack & Queue

class LinkedList(object):
    def __init__(self):
        self.head = None
        self.size = 0

    def append(self, val: int, index: int):
        if index < 0 or index > self.size:
            raise IndexError("Index out of bounds")

        if self.isEmpty():
            if index != 0:
                raise IndexError("Index out of bounds")
            self.head = ListNode(val)
            self.size += 1
            return
        dummy = ListNode(-1)
        dummy.next = self.head
        (pre, cur) = (dummy, self.head)
        for i in range(index):
            pre = pre.next
            cur = cur.next
        newNode = ListNode(val)
        newNode.next = cur
        pre.next = newNode
        self.head = dummy.next
        self.size += 1


    def remove(self, index: int):
        if self.isEmpty():
            raise Exception("The LinkedList is empty!")
        if index >= self.size:
            raise Exception("Invalid index!")
        if index == 0:  # Special case: removing the head
            result = self.head.val
            self.head = self.head.next  # Correctly handles a single-node case
            self.size -= 1
            return result
        dummy = ListNode(-1)
        dummy.next = self.head
        (cur, pre) = (self.head, dummy)
        for i in range(index):
            pre = pre.next
            cur = cur.next
        result = cur.val
        pre.next = cur.next
        self.size -= 1
        self.head = dummy.next
        return result

    def isEmpty(self):
        return self.size == 0

    def print(self):
        if self.head is None:
            print("None")
        cur = self.head
        while cur is not None:
            print(cur.val, end = ' ')
            cur = cur.next

class Queue(object):
    def __init__(self):
        self.ll = LinkedList()

    def offer(self, val: int):
        self.ll.append(val, self.ll.size)

    def poll(self):
        if self.isEmpty():
            raise Exception("Queue is empty!")
        return self.ll.remove(0)

    def peek(self):
        if self.isEmpty():
            raise Exception("Queue is empty!")
        return self.ll.head.val

    def isEmpty(self):
        return self.ll.isEmpty()

    def size(self):
        return self.ll.size


class Stack(object):
    def __init__(self):
        self.ll = LinkedList()

    def offerFirst(self, val):
        self.ll.append(val, 0)

    def pollFirst(self):
        if self.ll.isEmpty():
            raise Exception("Stack is empty!")
        return self.ll.remove(0)

    def peekFirst(self):
        if self.ll.isEmpty():
            raise Exception("Stack is empty!")
        return self.ll.head.val

    def isEmpty(self):
        return self.ll.isEmpty()

    def size(self):
        return self.ll.size

    def printStack(self):
        self.ll.print()

def isStackPairwiseConsecutive(s: Stack):
    temporaryStack = Stack()
    result = True
    while s.size() != 0:
        val_1 = s.pollFirst()
        if s.size() >= 1:
            val_2 = s.pollFirst()
            if abs(val_1 - val_2) != 1:
                result = False
            temporaryStack.offerFirst(val_1)
            temporaryStack.offerFirst(val_2)
        else:
            temporaryStack.offerFirst(val_1)
    while not temporaryStack.isEmpty():
        s.offerFirst(temporaryStack.pollFirst())
    return result

def reverseQueue(q: Queue):
    s = Stack()
    while not q.isEmpty():
        s.offerFirst(q.poll())
    while not s.isEmpty():
        q.offer(s.pollFirst())

def removeUntil(s: Stack, val: int):
    while True:
        cur = s.peekFirst()
        if cur != val:
            s.pollFirst()
        else:
            break

def recursiveReverse(q: Queue):
    if q.isEmpty():
        return
    cur = q.poll()
    recursiveReverse(q)
    q.offer(cur)

def palindrome(str: str):
    s = Stack()
    length = len(str)
    for i in range(length // 2):
        s.offerFirst(str[i])
    if length % 2 == 0:
        startIndex = length // 2
    else:
        startIndex = (length + 1) // 2
    for i in range(startIndex, length):
        cur = s.pollFirst()
        if cur != str[i]:
            return False
    return True

def sortStack(s: Stack):
    temporaryStack = Stack()
    while not s.isEmpty():
        numToBeSorted = s.size()
        min = float('inf')
        count = 0
        while not s.isEmpty():
            cur = s.pollFirst()
            if cur < min:
                min = cur
                count = 1
            elif cur == min:
                count += 1
            temporaryStack.offerFirst(cur)
        for i in range(numToBeSorted):
            cur = temporaryStack.pollFirst()
            if cur != min:
                s.offerFirst(cur)
        for i in range(count):
            temporaryStack.offerFirst(min)
    while not temporaryStack.isEmpty():
        s.offerFirst(temporaryStack.pollFirst())


# Part 3: Binary Tree

class TreeNode(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def identical(root_1: TreeNode, root_2: TreeNode):
    if root_1 is None and root_2 is None:
        return True
    elif root_1 is None or root_2 is None:
        return False
    elif root_1.val != root_2.val:
        return False
    return identical(root_1.left, root_2.left) and identical(root_1.right, root_2.right)

def maxHeight(root: TreeNode):
    if root is None:
        return 0
    leftResult = maxHeight(root.left)
    rightResult = maxHeight(root.right)
    return max(leftResult, rightResult) + 1

def countOneChildNodes(root: TreeNode):
    if root is None:
        return 0
    rootHasOneChild = 0
    if (root.left is None and root.right is not None) or (root.left is not None and root.right is None):
        rootHasOneChild = 1
    return countOneChildNodes(root.left) + countOneChildNodes(root.right) + rootHasOneChild

def sumOfOddNodes(root: TreeNode):
    if root is None:
        return 0
    leftResult = sumOfOddNodes(root.left)
    rightResult = sumOfOddNodes(root.right)
    if root.val % 2 == 1:
        return leftResult + rightResult + 1
    else:
        return leftResult + rightResult

def mirrorTree(root: TreeNode):
    if root is None:
        return
    (root.right, root.left) = (root.left, root.right)
    mirrorTree(root.left)
    mirrorTree(root.right)
    return root

def SmallestValue(root: TreeNode):
    if root is None:
        return float('inf')
    leftSmallest = SmallestValue(root.left)
    rightSmallest = SmallestValue(root.right)
    return min(leftSmallest, rightSmallest, root.val)

def printSmallestValues(root: TreeNode):
    q = Queue()
    q.offer(root)
    while not q.isEmpty():
        numOfCurrentLevel = q.size()
        minVal = float('inf')
        for i in range(numOfCurrentLevel):
            currentNode = q.poll()
            if currentNode.val < minVal:
                minVal = currentNode.val
            if currentNode.left is not None:
                q.offer(currentNode.left)
            if currentNode.right is not None:
                q.offer(currentNode.right)
        print(minVal, end = ' ')

def hasGreatGrandChild(root: TreeNode):
    return maxHeight(root) >= 4

def levelOrderTranversal(root: TreeNode):
    q = Queue()
    q.offer(root)
    while not q.isEmpty():
        currentNode = q.poll()
        if currentNode.left is not None:
            q.offer(currentNode.left)
        if currentNode.right is not None:
            q.offer(currentNode.right)
        print(currentNode.val, end = ' ')

def inOrderTranversalRecursive(root: TreeNode):
    if root is None:
        return
    inOrderTranversalRecursive(root.left)
    print(root.val, end = ' ')
    inOrderTranversalRecursive(root.right)

def preOrderTranversalIterative(root: TreeNode):
    s = Stack()
    s.offerFirst(root)
    while not s.isEmpty():
        currentNode = s.pollFirst()
        print(currentNode.val, end = ' ')
        if currentNode.right is not None:
            s.offerFirst(currentNode.right)
        if currentNode.left is not None:
            s.offerFirst(currentNode.left)

def inOrderTranversalIterative(root: TreeNode):
    s = Stack()
    cur = root
    while not s.isEmpty() or cur is not None:
        while cur is not None:
            s.offerFirst(cur)
            cur = cur.left
        cur = s.pollFirst()
        print(cur.val, end = ' ')
        cur = cur.right

def postOrderTranversalIterative1(root: TreeNode):
    if root is None:
        return
    pre = None
    s = Stack()
    s.offerFirst(root)
    while not s.isEmpty():
        cur = s.peekFirst()
        if pre is None or pre.left == cur or pre.right == cur:
            if cur.left is not None:
                s.offerFirst(cur.left)
            elif cur.right is not None:
                s.offerFirst(cur.right)
            else:
                print(cur.val, end=' ')
                s.pollFirst()
        elif pre == cur.left:
            if cur.right is not None:
                s.offerFirst(cur.right)
            else:
                print(cur.val, end=' ')
                s.pollFirst()
        else:
            print(cur.val, end=' ')
            s.pollFirst()
        pre = cur


def postOrderTranversalIterative2(root: TreeNode):
    s1 = Stack()
    s2 = Stack()
    s1.pollFirst(root)
    while not s1.isEmpty():
        currentNode = s1.pollFirst()
        s2.offerFirst(currentNode)
        if currentNode.left is not None:
            s1.offerFirst(currentNode.left)
        if currentNode.right is not None:
            s1.offerFirst(currentNode.right)
    while not s2.isEmpty():
        print(s2.pollFirst().val, end = ' ')

def isBST(root: TreeNode):
    minValue = sys.maxsize
    maxValue = -sys.maxsize - 1
    return isBSTHelper(root, minValue, maxValue)

def isBSTHelper(root: TreeNode, minValue, maxValue):
    if root is None:
        return True
    if not (minValue < root.val < maxValue):
        return False
    return isBSTHelper(root.left, minValue, root.val) and isBSTHelper(root.right, root.val, maxValue)

def removeBSTNodeIterative(root: TreeNode, target: int):
    (parent, cur) = (None, root)
    while cur is not None and cur.val != target:
        parent = cur
        if cur.val < target:
            cur = cur.right
        else:
            cur = cur.left
    if cur is None:
        return root
    if cur.left is None and cur.right is None:
        if parent is None:
            return None
        elif parent.val < cur.val:
            parent.right = None
        else:
            parent.left = None
    elif cur.left is None or cur.right is None:
        newChild = cur.left if cur.left is not None else cur.right
        if parent is None:
            return newChild
        replaceChild(parent, cur, newChild)
    else:
        parentOfSuccessor = cur
        successor = cur.right
        while successor.left is not None:
            parentOfSuccessor = successor
            successor = successor.left
        if parentOfSuccessor == cur:
            cur.val = successor.val
            cur.right = successor.right
        else:
            cur.val = successor.val
            replaceChild(parentOfSuccessor, successor, successor.right)
    return root

def replaceChild(parent: TreeNode, target: TreeNode, newChild: TreeNode):
    if parent.val < target.val:
        parent.right = newChild
    else:
        parent.left = newChild

def removeBSTNodeRecursive(root: TreeNode, target: int):
    if root is None:
        return None
    if root.val < target:
        root.right = removeBSTNodeRecursive(root.right, target)
    elif root.val > target:
        root.left = removeBSTNodeRecursive(root.left, target)
    else:
        if root.left is None and root.right is None:
            return None
        elif root.left is None:
            return root.right
        elif root.right is None:
            return root.left
        else:
            successor = root.right
            parentOfSuccessor = root
            while successor.left is not None:
                parentOfSuccessor = successor
                successor = successor.left
            root.val = successor.val
            if parentOfSuccessor == root:
                root.right = removeBSTNodeRecursive(root.right, successor.val)
            else:
                parentOfSuccessor.left = removeBSTNodeRecursive(parentOfSuccessor.left, successor.val)
    return root

def insertBSTNode(root: TreeNode, val: int):
    if root is None:
        return TreeNode(val)
    cur = root
    parent = None
    while cur is not None:
        parent = cur
        if cur.val < val:
            cur = cur.right
        else:
            cur = cur.left
    if parent.val > val:
        parent.left = TreeNode(val)
    else:
        parent.right = TreeNode(val)
    return root