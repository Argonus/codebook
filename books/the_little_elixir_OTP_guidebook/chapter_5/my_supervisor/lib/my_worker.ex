defmodule MySupervisor.MyWorker do
  def start_link do
    spawn(&loop/0)
  end

  defp loop do
    receive do
      :stop ->
        :ok

      msg ->
        IO.inspect(msg)
        loop()
    end
  end
end

{:ok, sup_pid} = MySupervisor.start_link([])
{:ok, child_spec} = MySupervisor.start_child(sup_pid, {MySupervisor.MyWorker, :start_link, []})
MySupervisor.which_children(sup_pid)

Process.info(sup_pid, :links)
Process.exit(child_spec, :crash)
