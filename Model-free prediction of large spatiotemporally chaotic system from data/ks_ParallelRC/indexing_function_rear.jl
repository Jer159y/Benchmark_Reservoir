function indexing_function_rear(chunk_begin, locality, num_inputs)
    chunk_begin = Int(chunk_begin)
    locality = Int(locality)
    num_inputs = Int(num_inputs)
    
    if chunk_begin - locality > 0
        return chunk_begin-locality : chunk_begin-1
    elseif chunk_begin - locality <= 0 && chunk_begin > 1
        i1 = mod(chunk_begin - locality, num_inputs)
        if i1 == 0; i1 = num_inputs; end # MATLAB mod vs Julia mod 0 handling check
        # Julia의 mod(x, y)는 결과가 [0, y-1]입니다. MATLAB은 1-based라 [1, y]처럼 쓰일 때가 많음.
        # 여기서는 논리적으로 뒤쪽에서 돌아오는 인덱스임.
        # 예: chunk_begin=2, loc=5, N=100 -> -3 -> mod(-3, 100) = 97.
        i1 = mod(chunk_begin - locality - 1, num_inputs) + 1
        return vcat(i1:num_inputs, 1:chunk_begin-1)
    elseif chunk_begin - locality <= 0 && chunk_begin == 1
        i1 = mod(chunk_begin - locality - 1, num_inputs) + 1
        return i1 : num_inputs
    else
        error("Indexing error in rear overlap")
    end
end