import turtle
mypen = turtle.Pen()
myscreen = turtle.Screen()
myscreen.tracer(0)
mypen.hideturtle()
mypen.penup()
mypen.goto(350, 0)
mypen.pendown()
mypen.pencolor("skyblue")
mypen.write("编程猫", font=("文泉驿正黑", 50))
i = 0
while i < 300:
    mypen.write("编程猫", font=("文泉驿正黑", 50))
    mypen.clear()
    mypen.backward(1)
    i = i+1
turtle.done()
