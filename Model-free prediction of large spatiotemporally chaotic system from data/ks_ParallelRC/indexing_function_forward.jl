function indexing_function_forward(chunk_end, locality, num_inputs)
    chunk_end = Int(chunk_end)
    locality = Int(locality)
    num_inputs = Int(num_inputs)
    
    if chunk_end + locality <= num_inputs
        return chunk_end+1 : chunk_end+locality
    elseif chunk_end + locality > num_inputs && chunk_end == num_inputs
        return 1 : mod(chunk_end + locality, num_inputs)
    elseif chunk_end + locality > num_inputs && chunk_end < num_inputs
        part1 = chunk_end+1 : num_inputs
        part2 = 1 : mod(chunk_end + locality, num_inputs)
        return vcat(part1, part2)
    else
        error("Indexing error in forward overlap")
    end
end