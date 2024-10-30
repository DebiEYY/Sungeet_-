# declare node
class Node:
    def __init__(self, data1, data2):
        self.data1= data1
        self.data2 = data2
        self.next = None


# declare circular list
class CircularList:
    def __init__(self):
        self.head = None

    def append(self, data1, data2):
        # Append a new node with data to the end of the circular linked list
        new_node = Node(data1, data2)
        if not self.head:
            # If the list is empty, make the new node point to itself
            new_node.next = new_node
            self.head = new_node
        else:
            current = self.head
            while current.next != self.head:
                # Traverse the list until the last node
                current = current.next
            # Make the last node point to the new node
            current.next = new_node
            # Make the new node point back to the head
            new_node.next = self.head

    def display(self):
        # Display the elements of the linked list
        current = self.head
        while current:
            # Traverse through each node and print its data
            print(current.note, end=" -> ")
            current = current.next
        # Print None to indicate the end of the linked list
        print("None")


class Chord:
    def __init__(self, name, degree):
        self.name = name
        self.degree = degree
        self.notes = []

    def add_note(self, note, pitch):
        self.notes.append((note, pitch))

    def display(self):
        print(f"name: {self.name}\n")
        print(f"degree: {self.degree}\n")
        for note, pitch in self.notes:
            print(f"note: {note}, pitch: {pitch}")
