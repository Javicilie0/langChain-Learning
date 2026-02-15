
from dotenv import load_dotenv


load_dotenv()
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage

todos = []


@tool
def add_Todo(task: str, priority: str) -> str:
    """Add new todo. Priority must be High, Medium or Low"""
    todo = {
        "task": task,
        "priority": priority,
        "done": False
    }
    todos.append(todo)
    return f"Added todo: {task} with priority {priority}"

@tool
def list_Todos() -> str:
    """List all todos"""
    if not todos:
        return "No todos yet."
    return "\n".join([f"{idx+1}. {todo['task']} (Priority: {todo['priority']}, Done: {todo['done']})" for idx, todo in enumerate(todos)])

@tool
def mark_done(index: int) -> str:
    """Mark a todo as done by its index (starting from 1)"""
    if index < 1 or index > len(todos):
        return "Invalid index."
    todos[index-1]["done"] = True
    return f"Marked todo #{index} as done."

@tool
def remove_Todo(index: int) -> str:
    """Remove a todo by its index (starting from 1)"""
    if index < 1 or index > len(todos):
        return "Invalid index."
    removed = todos.pop(index-1)
    return f"Removed todo: {removed['task']}"


tools = [add_Todo, list_Todos,mark_done,remove_Todo]

llm = ChatOllama(model="gpt-oss:20b", temperature=0)
llm_with_tools = llm.bind_tools(tools)




def run_agent(question: str) -> str:

    max_iterations = 5
    iteration = 0
    messages = [HumanMessage(content=question)]

    while iteration < max_iterations:
        iteration += 1
        print(f"[Iteration {iteration}]")
        
        ai_message = llm_with_tools.invoke(messages)
        
    
        if ai_message.tool_calls and len(ai_message.tool_calls) > 0:
            print("AI wants to use a tool...")
            messages.append(ai_message)
            
            for tool_call in ai_message.tool_calls:
                tool_name = tool_call["name"]
                args = tool_call["args"]
                tool_id = tool_call["id"]
                
                print(f"  Tool: {tool_name}")
                print(f"  Args: {args}")
                
                for t in tools:
                    if t.name == tool_name:
                        result = t.invoke(args)  
                        print(f"  Result: {result}")
                        
                        messages.append(ToolMessage(
                            content=str(result),
                            tool_call_id=tool_id 
                        ))
                        break
            
            continue  
        

        print(f"Final answer: {ai_message.content}")
        break

    print(f"\nTodos list: {todos}")


def main():
    print("Hello from langchain-learninggg!")
    
    while True:
        user_input = input("\nAsk a question or type 'exit' to quit: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        run_agent(user_input)


if __name__ == "__main__":
    main()
