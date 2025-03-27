# Sum random numbers in a loop of 100_000_000 iterations
@time let
    total = 0
    for i in 1:100_000_000
        total += rand(1:10)
    end
    println(total)
end
