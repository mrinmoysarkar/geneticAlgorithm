clear all;
close all;

for l=2:55
    m_min = 1;
    m_max = 100^100;
    m1_prev = 0;
    m2_prev = 0;
    count = 0;
    m_opt = 0;
    delm = 0;
    while 1
        m1 = ceil((m_max-m_min)/3) + 1;
        m2 = ceil((2*(m_max-m_min))/3) + 1;
        f1 = f(m1,l);
        f2 = f(m2,l);
        if (m2 - m1) == delm
            count = count + 1;
            m_opt = max(m2,m_opt);
        end
        if f1 > f2
            m_max = m2;
        else
            m_min = m1;
        end
        %fprintf('m1=%d  m2=%d\n',m1,m2);
        if count == 2
            fprintf('Optimal Population Size m* = %d when string length l = %d\n', m_opt,l);
            break;
        end
        delm = m2-m1;
    end
end